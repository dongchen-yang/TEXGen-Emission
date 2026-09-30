#!/usr/bin/env python3
"""
Inference script for specific sample IDs using a trained checkpoint.
Generates input, ground truth, and predicted emission maps.

This script matches the validation process from training:
- Uses EMA weights if enabled in config
- Disables autocast during inference
- Respects data_normalization setting from config
"""

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
    """Per-shape RNG seed for `(sha, seed)`. Order-, batch- and skip-independent.

    VERBATIM COPY of evaluation/newdata_eval/seedutil.py in the parent lightgen repo, duplicated
    because this repo is cloned standalone on cs-venus-05. Keep the copies byte-identical.
    """
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

    data_root/parquet_file override the hardcoded full-dataset paths (used by the newdata_eval
    harness to point at a staged native-somage root that also carries thumbnails/).
    """
    
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
    
    if config_path.exists():
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
    else:
        print(f"   Warning: Config not found at {config_path}, using default")
        cfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/lightgen_pointuv_256_batch32_pretrained.yaml"))
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"   ✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load data module
    print("\n2. Setting up data module...")
    from spuv.data.lightgen_uv import LightGenDataModule, LightGenDataset
    
    # Override to use full dataset (not filtered) to find all samples
    cfg_data = cfg.data.copy()
    # Use full dataset paths
    cfg_data.parquet_file = parquet_file or "/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet"
    cfg_data.data_root = data_root or "/localhome/dya78/code/lightgen/data/baked_uv"
    print(f"   Using dataset: {cfg_data.parquet_file}")
    print(f"   data_root: {cfg_data.data_root} (thumbnails from {cfg_data.data_root}/thumbnails/)")
    
    # Don't apply train/val/test filters
    cfg_data.test_indices = None
    
    # Create a custom dataset that only loads specific samples
    data_module = LightGenDataModule(cfg_data)
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
    from spuv.systems.lightgen_system import LightGenSystem
    
    # Create model from config first (avoids config merge issues)
    # Pass only the system config, not the full config
    model = LightGenSystem(cfg.system)
    
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
    
    all_results = []  # Store results for overall visualization
    
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
        #     (lightgen_system.py:247,255) -- discarded, but they advance the stream;
        #   * the initial noise, one CUDA draw (texgen_test.py:506);
        #   * hundreds of CPU torch.randperm calls per shape from shuffle_orders=True
        #     (texgen_network.py:621,766 -> ptv3_model_texgen.py:129,703). That flag is
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
            
            # Get thumbnail - load directly from baked_uv_local since data_root is different
            thumbnail_img = None
            thumbnail_path = f"/localhome/dya78/code/lightgen/data/baked_uv_local/thumbnails/{sample_id}.png"
            if os.path.exists(thumbnail_path):
                thumb_pil = Image.open(thumbnail_path).convert('RGB')
                thumbnail_img = np.array(thumb_pil).astype(np.float32) / 255.0  # [H, W, 3]
            elif 'thumbnail' in batch and batch['thumbnail'] is not None:
                # Fallback to batch thumbnail if available
                thumbnail = batch['thumbnail']  # [1, 1, H, W, 3]
                if thumbnail.dim() == 5:
                    thumbnail_img = thumbnail[0, 0].cpu().numpy()  # [H, W, 3]
                else:
                    thumbnail_img = thumbnail[0].cpu().numpy()
            
            # Save individual images
            sample_dir = output_dir / sample_id
            sample_dir.mkdir(exist_ok=True)
            
            Image.fromarray((albedo_img * 255).astype(np.uint8)).save(sample_dir / "input_albedo.png")
            Image.fromarray((gt_img * 255).astype(np.uint8)).save(sample_dir / "gt_emission.png")
            Image.fromarray((pred_img * 255).astype(np.uint8)).save(sample_dir / "pred_emission.png")
            Image.fromarray((mask_img * 255).astype(np.uint8)).save(sample_dir / "mask.png")
            
            if thumbnail_img is not None:
                Image.fromarray((thumbnail_img * 255).astype(np.uint8)).save(sample_dir / "thumbnail.png")
            
            # Store for overall visualization
            all_results.append({
                'sample_id': sample_id,
                'thumbnail': thumbnail_img,
                'albedo': albedo_img,
                'gt_emission': gt_img,
                'pred_emission': pred_img,
            })
            
            # Create comparison image (3 columns: input, gt, pred)
            h, w = pred_img.shape[:2]
            comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
            comparison[:, :w] = (albedo_img * 255).astype(np.uint8)
            comparison[:, w:2*w] = (gt_img * 255).astype(np.uint8)
            comparison[:, 2*w:] = (pred_img * 255).astype(np.uint8)
            
            Image.fromarray(comparison).save(sample_dir / "comparison.png")
            
            print(f"      ✓ Saved to {sample_dir}/")
            print(f"         - input_albedo.png")
            print(f"         - gt_emission.png")
            print(f"         - pred_emission.png")
            if thumbnail_img is not None:
                print(f"         - thumbnail.png")
            print(f"         - comparison.png (input | gt | pred)")
            
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
    
    # Create overall visualization grid
    if all_results:
        print("\n8. Creating overall visualization...")
        create_overall_visualization(all_results, output_dir)

    print("\n" + "=" * 80)
    print("✓ Inference Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Settings: EMA={use_ema and val_with_ema}, data_normalization={data_normalization}")
    print("=" * 80)


def create_overall_visualization(all_results, output_dir):
    """Create a grid visualization with all samples.
    
    Layout: Each row is one sample
    Columns: Thumbnail | Albedo | GT Emission | Pred Emission
    """
    import cv2
    
    n_samples = len(all_results)
    
    # Get dimensions from first sample
    h, w = all_results[0]['albedo'].shape[:2]
    
    # Spacing between columns and rows
    col_spacing = 10
    row_spacing = 10
    
    # Get thumbnail dimensions (keep original aspect ratio, scale to match row height)
    thumb_h = h
    if all_results[0]['thumbnail'] is not None:
        orig_thumb = all_results[0]['thumbnail']
        orig_h, orig_w = orig_thumb.shape[:2]
        aspect_ratio = orig_w / orig_h
        thumb_w = int(thumb_h * aspect_ratio)
    else:
        thumb_w = h  # Square fallback
    
    # Column widths: thumbnail (original aspect) + 3 UV maps
    col_widths = [thumb_w, w, w, w]
    total_width = sum(col_widths) + col_spacing * 3  # 3 gaps between 4 columns
    total_height = n_samples * h + row_spacing * (n_samples - 1)  # gaps between rows
    
    # Add header row
    header_height = 40
    total_height += header_height
    
    # Create white canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Add header labels
    from PIL import ImageDraw, ImageFont
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    headers = ["Thumbnail", "Albedo (Input)", "GT Emission", "Pred Emission"]
    x_positions = [
        0,
        col_widths[0] + col_spacing,
        col_widths[0] + col_spacing + col_widths[1] + col_spacing,
        col_widths[0] + col_spacing + col_widths[1] + col_spacing + col_widths[2] + col_spacing
    ]
    
    for header, x_pos, col_w in zip(headers, x_positions, col_widths):
        # Center text in column
        text_bbox = draw.textbbox((0, 0), header, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_pos + (col_w - text_width) // 2
        draw.text((text_x, 10), header, fill=(0, 0, 0), font=font)
    
    canvas = np.array(canvas_pil)
    
    # Fill in each row
    for i, result in enumerate(all_results):
        row_start = header_height + i * (h + row_spacing)
        
        # Column 0: Thumbnail (keep aspect ratio, scale height to h)
        if result['thumbnail'] is not None:
            thumb = result['thumbnail']
            orig_h, orig_w = thumb.shape[:2]
            aspect_ratio = orig_w / orig_h
            new_w = int(h * aspect_ratio)
            thumb_resized = cv2.resize(thumb, (new_w, h), interpolation=cv2.INTER_LINEAR)
            canvas[row_start:row_start + h, 0:new_w] = (thumb_resized * 255).astype(np.uint8)
        
        # Column 1: Albedo
        col_start = col_widths[0] + col_spacing
        canvas[row_start:row_start + h, col_start:col_start + w] = (result['albedo'] * 255).astype(np.uint8)
        
        # Column 2: GT Emission
        col_start = col_widths[0] + col_spacing + col_widths[1] + col_spacing
        canvas[row_start:row_start + h, col_start:col_start + w] = (result['gt_emission'] * 255).astype(np.uint8)
        
        # Column 3: Pred Emission
        col_start = col_widths[0] + col_spacing + col_widths[1] + col_spacing + col_widths[2] + col_spacing
        canvas[row_start:row_start + h, col_start:col_start + w] = (result['pred_emission'] * 255).astype(np.uint8)
    
    # Save overall visualization
    output_path = output_dir / "overall_visualization.png"
    Image.fromarray(canvas).save(output_path)
    print(f"   ✓ Saved overall visualization to: {output_path}")


if __name__ == "__main__":
    checkpoint_path = "/localhome/dya78/code/lightgen/TEXGen/ckpts/epoch=599-step=17400.ckpt"
    
    # Test set samples
    sample_ids = [
        "008ac92377d547c391a83f96e485c313",
        "0d66475a413b4c6288178ad94251677f",
        "0ee07f5656fc41bf9f36bc19e6605017",
        "20a021a4f4eb4bc6a920d018ef2d02a2",
        "4532362cb3764e0e90547d886a37f52a",
        "6cfba09348b64230b274026e175710ed",
        "9c7467268bf34ada8e57429736dfba19",
        "fff48e914c4847a08660b9e08b1b733c",
        "21170b1a94664f2fae7c00c4e2c25577",
        "f42e52eb50c041cfb4e29310e4a95e33",
    ]
    
    output_dir = "inference_outputs_epoch599"
    
    inference_samples(checkpoint_path, sample_ids, output_dir)
