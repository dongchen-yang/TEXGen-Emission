#!/usr/bin/env python3
"""
Inference script for specific sample IDs using a trained checkpoint.
Generates input, ground truth, and predicted emission maps.

    python inference_specific_samples.py --ckpt <run>/ckpts/<file>.ckpt --data_root <root> \
        --parquet <root>/<index>.parquet --shas_file <shapes.txt> --out_dir <out> [--seed 0]

The run's config is read beside the checkpoint (<run>/configs/parsed.yaml; a published run's config
and checkpoint go through scripts/convert_config.py first). Writes <out>/<sha>/{pred_emission,
gt_emission,mask,input_albedo}.png.

This script matches the validation process from training:
- Uses EMA weights if enabled in config
- Disables autocast during inference
- Respects data_normalization setting from config
"""

import argparse
import hashlib
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

# Add TEXGen to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sample_seed(sha: str, seed: int) -> int:
    """Per-shape RNG seed for `(sha, seed)`. Order-, batch- and skip-independent."""
    return int.from_bytes(hashlib.sha256(f"{sha}:{seed}".encode()).digest()[:8], "big")


def find_sample_index(sample_id, parquet_file):
    """Find the positional index of a sample ID in the parquet file."""
    df = pd.read_parquet(parquet_file)
    df_success = df[df['success'] == True].copy()
    
    if sample_id in df_success.index:
        position = list(df_success.index).index(sample_id)
        return position
    else:
        return None

def inference_samples(checkpoint_path, sample_ids, output_dir, data_root=None,
                      parquet_file=None, seed=0):
    """Run inference on specific samples and save results.

    data_root (somage.npz per shape and thumbnails/) and parquet_file (its index) are required.
    """
    if not data_root or not parquet_file:
        raise ValueError("inference_samples needs data_root and parquet_file")
    
    print("=" * 80)
    print("LightGen Inference for Specific Samples")
    print("=" * 80)
    
    # Load checkpoint and config
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    
    # First try to load config from checkpoint directory
    # The structure is: .../run_dir/ckpts/checkpoint.ckpt or .../run_dir/nested/ckpts/checkpoint.ckpt
    # Try both locations
    ckpt_path_obj = Path(checkpoint_path)
    config_path = ckpt_path_obj.parent.parent / "configs" / "parsed.yaml"
    if not config_path.exists():
        # Try going up one more level for nested structure
        config_path = ckpt_path_obj.parent.parent.parent / "configs" / "parsed.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"no configs/parsed.yaml beside {checkpoint_path}")
    print(f"   Loading config from: {config_path}")
    full_cfg = OmegaConf.load(config_path)
    # Extract only the keys needed for the model (avoid metadata keys)
    cfg = OmegaConf.create({
        'data_cls': full_cfg.get('data_cls'),
        'data': full_cfg.get('data'),
        'system_cls': full_cfg.get('system_cls'),
        'system': full_cfg.get('system'),
        'trainer': full_cfg.get('trainer', {}),
        'checkpoint': full_cfg.get('checkpoint', {}),
    })

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"   ✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load data module
    print("\n2. Setting up data module...")
    from spuv.data.mesh_uv import ObjaverseDataModule
    
    # The given data, without the run's split filters, so that any listed shape can be found
    cfg_data = cfg.data.copy()
    cfg_data.parquet_file = parquet_file
    cfg_data.data_root = data_root
    print(f"   Using dataset: {cfg_data.parquet_file}")
    print(f"   data_root: {cfg_data.data_root} (thumbnails from {cfg_data.data_root}/thumbnails/)")
    
    # Don't apply train/val/test filters
    cfg_data.test_indices = None
    
    # Create a custom dataset that only loads specific samples
    data_module = ObjaverseDataModule(cfg_data)
    data_module.setup('test')
    
    # Find indices for our sample IDs
    parquet_file = cfg_data.parquet_file
    print(f"\n3. Finding sample indices in: {parquet_file}")
    
    sample_indices = {}
    for sample_id in sample_ids:
        idx = find_sample_index(sample_id, parquet_file)
        if idx is not None:
            sample_indices[sample_id] = idx
            print(f"   ✓ Found {sample_id} at index {idx}")
        else:
            print(f"   ✗ Sample {sample_id} not found in dataset")
    
    if not sample_indices:
        print("\nError: No valid samples found!")
        return
    
    # Load the model
    print("\n4. Loading model...")
    from spuv.systems.texgen_train import TEXGenDiffusion
    
    # Create model from config first (avoids config merge issues)
    # Pass only the system config, not the full config
    model = TEXGenDiffusion(cfg.system)
    
    # Load weights from checkpoint
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   ✓ Model loaded on {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n5. Output directory: {output_dir}")
    
    # Check config settings
    use_ema = cfg.system.get('use_ema', True)
    val_with_ema = cfg.system.get('val_with_ema', True)
    data_normalization = cfg.system.get('data_normalization', True)
    
    print(f"\n6. Config settings:")
    print(f"   - use_ema: {use_ema}")
    print(f"   - val_with_ema: {val_with_ema}")
    print(f"   - data_normalization: {data_normalization}")
    
    # Run inference for each sample
    print("\n7. Running inference...")
    
    for sample_id, idx in sample_indices.items():
        print(f"\n   Processing: {sample_id} (index {idx})")
        
        # Get the sample
        dataset = data_module.test_dataset
        sample = dataset[idx]
        
        # Create batch (add batch dimension)
        batch = {}
        for key, value in sample.items():
            if key == 'thumbnail' and value is None:
                # Skip thumbnail if it's None - system will use albedo fallback
                continue
            elif isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(device)
            elif isinstance(value, dict):
                batch[key] = value
            elif key == 'thumbnail' and isinstance(value, torch.Tensor):
                # Thumbnail should be moved to device but keep its shape
                batch[key] = value.to(device)
            elif isinstance(value, (list, tuple)):
                batch[key] = [value]
            else:
                batch[key] = [value]

        # GLOBAL seed, and it must land HERE -- before test_pipeline, not inside it.
        # Three RNG consumers sit downstream and two of them are easy to miss:
        #   * prepare_diffusion_data draws and DISCARDS two CUDA tensors
        #     (texgen_test.py:67,75) -- discarded, but they advance the stream;
        #   * the initial noise, one CUDA draw (texgen_test.py:561);
        #   * hundreds of CPU torch.randperm calls per shape from shuffle_orders=True
        #     (texgen_network.py:647,792 -> ptv3_model_texgen.py:129,703). That flag is
        #     hardcoded and NOT gated on self.training -- the only self.training guard in
        #     either file is attention dropout -- and the permutation CHANGES THE OUTPUT by
        #     reordering the space-filling-curve serialization the attention windows use.
        # torch.manual_seed seeds CPU *and* all CUDA devices; a torch.Generator(device="cuda")
        # would leave every one of those randperm calls unseeded and seed 0 would not
        # reproduce itself. Do not "clean this up" into a Generator.
        torch.manual_seed(sample_seed(sample_id, seed))

        # Run inference - matching validation process exactly
        with torch.no_grad():
            # Disable autocast (matching validation)
            with torch.cuda.amp.autocast(enabled=False):
                # Use EMA weights if enabled (matching validation)
                if use_ema and val_with_ema:
                    print(f"      Using EMA weights for inference")
                    with model.ema_scope("Inference with ema weights"):
                        texture_map_outputs = model.test_pipeline(batch)
                else:
                    print(f"      Using regular weights for inference")
                    texture_map_outputs = model.test_pipeline(batch)
            
            # Extract results
            pred_emission = texture_map_outputs["pred_x0"][0]  # [3, H, W]
            gt_emission = texture_map_outputs["gt_x0"][0]  # [3, H, W]
            mask_map = texture_map_outputs["mask_map"][0, 0]  # [H, W]
            
            # Get input (albedo map)
            albedo_map = batch["albedo_map"][0]  # [3, H, W]
            
            # Denormalize based on config (matching validation)
            if data_normalization:
                pred_emission = (pred_emission * 0.5 + 0.5).clamp(0, 1)
                gt_emission = (gt_emission * 0.5 + 0.5).clamp(0, 1)
            else:
                pred_emission = pred_emission.clamp(0, 1)
                gt_emission = gt_emission.clamp(0, 1)
            # albedo is already in [0, 1] range
            
            # Apply mask
            pred_emission = pred_emission * mask_map.unsqueeze(0)
            gt_emission = gt_emission * mask_map.unsqueeze(0)
            albedo_map = albedo_map * mask_map.unsqueeze(0)
            
            # Convert to numpy and transpose to HWC
            pred_img = pred_emission.cpu().permute(1, 2, 0).numpy()
            gt_img = gt_emission.cpu().permute(1, 2, 0).numpy()
            albedo_img = albedo_map.cpu().permute(1, 2, 0).numpy()
            mask_img = mask_map.cpu().numpy()

            # Save individual images
            sample_dir = output_dir / sample_id
            sample_dir.mkdir(exist_ok=True)

            Image.fromarray((albedo_img * 255).astype(np.uint8)).save(sample_dir / "input_albedo.png")
            Image.fromarray((gt_img * 255).astype(np.uint8)).save(sample_dir / "gt_emission.png")
            Image.fromarray((pred_img * 255).astype(np.uint8)).save(sample_dir / "pred_emission.png")
            Image.fromarray((mask_img * 255).astype(np.uint8)).save(sample_dir / "mask.png")
            print(f"      ✓ Saved input_albedo, gt_emission, pred_emission, mask to {sample_dir}/")

            # Compute metrics
            from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            
            # Expand dims for metrics
            pred_batch = pred_emission.unsqueeze(0)
            gt_batch = gt_emission.unsqueeze(0)
            mask_batch = mask_map.unsqueeze(0)
            
            # Compute metrics in masked regions
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            
            # Apply mask for metrics
            pred_masked = pred_batch * mask_batch.unsqueeze(1)
            gt_masked = gt_batch * mask_batch.unsqueeze(1)
            
            psnr = psnr_metric(pred_masked, gt_masked)
            ssim = ssim_metric(pred_masked, gt_masked)
            
            print(f"         PSNR: {psnr.item():.2f} dB")
            print(f"         SSIM: {ssim.item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Inference Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Settings: EMA={use_ema and val_with_ema}, data_normalization={data_normalization}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Inference with a trained checkpoint on listed shapes.")
    ap.add_argument("--ckpt", required=True, help="<run>/ckpts/<file>.ckpt; the config is <run>/configs/parsed.yaml")
    ap.add_argument("--data_root", required=True, help="the shapes' <ditem_dir>/somage.npz and thumbnails/<sha>.png")
    ap.add_argument("--parquet", required=True, help="the data root's index (rows with success == True)")
    ap.add_argument("--shas_file", required=True, help="one shape id per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="per-shape seeds derive from (sha, seed)")
    a = ap.parse_args()
    shas = [l.strip() for l in open(a.shas_file) if l.strip()]
    inference_samples(a.ckpt, shas, a.out_dir, data_root=a.data_root, parquet_file=a.parquet, seed=a.seed)


if __name__ == "__main__":
    main()
