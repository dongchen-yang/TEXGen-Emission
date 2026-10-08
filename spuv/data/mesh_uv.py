import json
import math
import os
import random
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transform
import torchvision.transforms.functional as TF
from PIL import Image
import imageio
import cv2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SequentialSampler

import spuv
from spuv.utils.config import parse_structured
from spuv.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from spuv.utils.typing import *
from spuv.utils.mesh_utils import *
from spuv.data.camera_strategy import camera_functions


def _parse_scene_list(scene_path):
    data = []
    with open(scene_path, 'r') as file:
        for line in file:
            try:
                # Parse each line as a JSON object
                json_data = json.loads(line.strip())
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line}")

    return data

def _parse_split_indices(split_path, split):
    # {"train"|"val"|"test": {"indices": [...]}}: positions into the parquet's success rows
    with open(split_path, 'r') as file:
        return json.load(file).get(split, {}).get('indices', [])

def file_from_uv_dir(uv_dir, mes_name="model.obj", uv_name="model.png"):
    mesh_file_path = os.path.join(uv_dir, mes_name)
    uv_image_path = os.path.join(uv_dir, uv_name)

    return mesh_file_path, uv_image_path


def decode_uint16_to_float(data, min_val=-2.0, max_val=2.0):
    """Decode uint16 data to float in range [min_val, max_val]"""
    return (data.astype(np.float32) / 65535.0) * (max_val - min_val) + min_val


def decode_uint8_to_float(data):
    """Decode uint8 data to float in range [0, 1]"""
    return data.astype(np.float32) / 255.0


# The atlas channels in somage.npz: name -> (decoder to float, resize mode). Nearest for the
# occupancy mask, bilinear for continuous values.
UV_CHANNELS = {
    "occupancy": (lambda a: a, "nearest"),
    "position": (decode_uint16_to_float, "bilinear"),  # [-2, 2]
    "color": (decode_uint8_to_float, "bilinear"),  # albedo
    "metal": (decode_uint8_to_float, "bilinear"),
    "rough": (decode_uint8_to_float, "bilinear"),
    "alpha": (decode_uint8_to_float, "bilinear"),  # glTF opacity
    "emission_color": (decode_uint8_to_float, "bilinear"),
}


@dataclass
class ObjaverseDataModuleConfig:
    data_root: str = ""
    parquet_file: str = ""
    scene_list: str = ""
    eval_scene_list: str = ""
    repeat: int = 1  # for debugging purpose
    camera_strategy: str = "strategy_1"
    eval_camera_strategy: str = "strategy_1"
    height: int = 128
    width: int = 128
    cond_views: int = 1
    sup_views: int = 4
    uv_height: int = 256
    uv_width: int = 256
    uv_channel: int = 3

    # A JSON split file (_parse_split_indices); None keeps every row
    train_indices: Optional[str] = None
    val_indices: Optional[str] = None
    test_indices: Optional[str] = None

    batch_size: int = 1
    num_workers: int = 4

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_cond_views: int = 1
    eval_sup_views: int = 4

    vertex_transformation: bool = False
    mesh_name: str = "model.obj"
    uv_name: str = "model.png"
    use_precomputed_clip: bool = False  # Load precomputed CLIP embeddings instead of raw thumbnails


class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.split = split

        # Load sample list from parquet file (much faster than scanning!)
        self.all_samples = self._load_from_parquet()

        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_samples = [
                self.all_samples[i] for i in _parse_split_indices(self.cfg.train_indices, "train")
            ]
            self.all_samples = self.all_samples * self.cfg.repeat
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_samples = [
                self.all_samples[i] for i in _parse_split_indices(self.cfg.val_indices, "val")
            ]
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_samples = [
                self.all_samples[i] for i in _parse_split_indices(self.cfg.test_indices, "test")
            ]

    def _load_from_parquet(self):
        """Load sample list from parquet file (much faster than scanning)"""
        if not self.cfg.data_root or not self.cfg.parquet_file:
            raise ValueError("data.data_root and data.parquet_file must be set in the config")

        # Read parquet file
        df = pd.read_parquet(self.cfg.parquet_file)
        
        # Filter for successful samples only
        if 'success' in df.columns:
            df = df[df['success'] == True]
        
        samples = []
        data_root = self.cfg.data_root
        
        # Convert to sample list
        for sample_id, row in df.iterrows():
            # Get relative path from ditem_dir column
            ditem_dir = row.get('ditem_dir', None)
            if ditem_dir is None:
                # Fallback: construct from sample_id
                ditem_dir = f"{sample_id[:3]}-{sample_id[3:6]}/{sample_id}"
            
            sample_path = os.path.join(data_root, ditem_dir)
            npz_file = os.path.join(sample_path, "somage.npz")
            
            samples.append({
                "sample_id": sample_id,
                "path": sample_path,
                "npz_file": npz_file,
            })

        return samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        try:
            return self.try_get_item(index)
        except Exception as e:
            print(f"Failed to load {index}: {e}")
            return None

    def vertex_transform(self, mesh):
        pre_vertices = mesh['v_pos']

        vertices = torch.clone(pre_vertices)
        vertices[:, 1] = -pre_vertices[:, 2]  # -z --> y
        vertices[:, 2] = pre_vertices[:, 1]  # y --> z

        bounding_box_max = vertices.max(0)[0]
        bounding_box_min = vertices.min(0)[0]
        mesh_scale = 1.0
        scale = mesh_scale / ((bounding_box_max - bounding_box_min).max() + 1e-6)
        center_offset = (bounding_box_max + bounding_box_min) * 0.5
        vertices = (vertices - center_offset) * scale

        mesh['v_pos'] = vertices

        return mesh

    def try_get_item(self, index):
        sample_info = self.all_samples[index]
        sample_id = sample_info["sample_id"]
        npz_file = sample_info["npz_file"]
        
        # Load every atlas channel: decode to float, [H, W, C] -> [C, H, W]
        with np.load(npz_file) as npz_data:
            maps = {name: torch.from_numpy(decode(npz_data[name])).float().permute(2, 0, 1)
                    for name, (decode, _) in UV_CHANNELS.items()}

        # Load CLIP conditioning: either precomputed embedding or raw thumbnail
        thumbnail = None
        clip_image_embedding = None
        if self.cfg.use_precomputed_clip:
            # Load precomputed CLIP image embedding
            emb_path = os.path.join(self.cfg.data_root, "clip_embeddings", f"{sample_id}.pt")
            if os.path.exists(emb_path):
                clip_image_embedding = torch.load(emb_path, weights_only=True)  # [768]
        else:
            # Load raw thumbnail for online CLIP encoding
            thumbnail_path = os.path.join(self.cfg.data_root, "thumbnails", f"{sample_id}.png")
            if os.path.exists(thumbnail_path):
                with Image.open(thumbnail_path) as thumbnail_pil:
                    thumbnail_img = thumbnail_pil.convert('RGB')
                    thumbnail_img = TF.resize(thumbnail_img, [224, 224], interpolation=TF.InterpolationMode.BILINEAR)
                    thumbnail = torch.from_numpy(np.array(thumbnail_img)).float() / 255.0  # [224, 224, 3]
                    thumbnail = thumbnail.unsqueeze(0)  # [1, 224, 224, 3]
        
        # GT emission mask: binary mask where any emission channel > threshold (in [0,1] space)
        # Multiply by occupancy so regions outside UV islands are always 0
        gt_emission_mask = ((maps["emission_color"].max(dim=0, keepdim=True)[0] > 0.001) * maps["occupancy"]).float()

        # Normalize emission_color to [-1, 1] for diffusion
        maps["emission_color"] = maps["emission_color"] * 2.0 - 1.0

        # Resize UV data to the configured resolution if needed
        current_h, current_w = maps["occupancy"].shape[-2:]
        target_h, target_w = self.cfg.uv_height, self.cfg.uv_width

        if current_h != target_h or current_w != target_w:
            size = (target_h, target_w)
            maps = {name: F.interpolate(maps[name].unsqueeze(0), size=size, mode=mode,
                                        align_corners=False if mode == "bilinear" else None).squeeze(0)
                    for name, (_, mode) in UV_CHANNELS.items()}
            gt_emission_mask = F.interpolate(gt_emission_mask.unsqueeze(0), size=size, mode="nearest").squeeze(0)

        if self.split == "train":
            height, width = self.cfg.height, self.cfg.width
        else:
            height, width = self.cfg.eval_height, self.cfg.eval_width

        return_dict = {
            "scene_id": sample_id,
            "index": index,
            "height": height,
            "width": width,
            "uv_height": self.cfg.uv_height,
            "uv_width": self.cfg.uv_width,
            
            # UV space data (resized to match configured resolution)
            "position_map": maps["position"],  # [3, H, W]
            "albedo_map": maps["color"],  # [3, H, W]
            "metal_map": maps["metal"],  # [1, H, W]
            "rough_map": maps["rough"],  # [1, H, W]
            "alpha_map": maps["alpha"],  # [1, H, W]
            "mask_map": maps["occupancy"],  # [1, H, W]
            "gt_emission": maps["emission_color"],  # [3, H, W], normalized to [-1, 1]
            "gt_emission_mask": gt_emission_mask,  # [1, H, W], binary mask where emission > 0
            
            # CLIP conditioning (one of these will be non-None)
            "thumbnail": thumbnail,  # [1, 224, 224, 3] or None (raw image for online encoding)
            "clip_image_embedding": clip_image_embedding,  # [768] or None (precomputed)
            
        }

        return return_dict

    # Items carry None for absent inputs (e.g. thumbnail when the CLIP embedding is precomputed),
    # which default_collate cannot batch: stack tensors, keep None, list everything else.
    def collate(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        collated = {}
        for key, first in batch[0].items():
            if first is None:
                collated[key] = None
            elif isinstance(first, torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                collated[key] = [item[key] for item in batch]

        return collated


class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            # self.test_dataset = ObjaverseOrbitDataset(self.cfg, "test")
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
        )

    # Val and test pass an explicit sampler so that under DDP every rank evaluates the full set
    # (Lightning would otherwise shard it); train keeps Lightning's sharding.
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
            sampler=SequentialSampler(self.val_dataset),
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
            sampler=SequentialSampler(self.test_dataset),
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = {
        "scene_list": "",
        "eval_scene_list": "",
        "test_indices": [0, 100],
        "uv_name": "",
        "height": 512,
        "width": 512,
        "batch_size": 1,
        "num_workers": 0,
    }

    dataset = ObjaverseDataModule(conf)
    dataset.setup()
    dataloader = dataset.test_dataloader()

    for batch in dataloader:
        breakpoint()

    breakpoint()
