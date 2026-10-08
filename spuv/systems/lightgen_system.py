from dataclasses import dataclass, field
import math
import random
from contextlib import contextmanager
import colorsys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import profiler
from torchvision.transforms import v2 as transform

import spuv
from spuv.utils.misc import get_device
from spuv.systems.texgen_test import TEXGenDiffusion
from spuv.systems.texgen_base import LossConfig
from spuv.utils.ops import binary_cross_entropy, get_plucker_rays
from spuv.utils.typing import *
from spuv.models.lpips import LPIPS
from spuv.utils.misc import time_recorder as tr
from spuv.utils.snr_utils import compute_snr_from_scheduler, get_weights_from_timesteps
from spuv.models.perceptual_loss import VGGPerceptualLoss
from spuv.models.renderers.rasterize import NVDiffRasterizerContext
from spuv.utils.mesh_utils import uv_padding
from spuv.utils.nvdiffrast_utils import *
from spuv.utils.lit_ema import LitEma
from spuv.utils.image_metrics import SSIM, PSNR
from spuv.utils.memory_tracker import log_memory, set_baseline


class LightGenSystem(TEXGenDiffusion):
    """
    LightGen System for generating emission maps from geometry and material properties.
    Inherits from texgen_test.TEXGenDiffusion to match original TEXGen exactly (with Flow Matching).
    
    Input: geometry (position, normal), material (albedo, roughness, metallic)
    Output: emission map in UV space
    """
    
    @dataclass
    class Config(TEXGenDiffusion.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        import time
        t0 = time.time()
        print(f"[stage] LightGenSystem.configure() — building backbone via super() ...", flush=True)
        super().configure()
        print(f"[stage] backbone built ({time.time()-t0:.1f}s)", flush=True)
        # Initialize image tokenizer for conditioning (CLIP — may pull from HF cache).
        t1 = time.time()
        print(f"[stage] building image tokenizer ({self.cfg.image_tokenizer_cls}) ...", flush=True)
        self.image_tokenizer = spuv.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        print(f"[stage] image tokenizer ready ({time.time()-t1:.1f}s)", flush=True)
        
        # Initialize memory tracker for debugging OOM issues
        from spuv.utils.memory_tracker import init_tracker
        # Log memory every step for first few epochs, then every 5 steps
        self.memory_tracker = init_tracker(enabled=True, log_interval=1)
        spuv.info("[MEMORY] Memory tracking enabled for OOM debugging")
        
    def prepare_condition_info(self, batch):
        """
        Prepare conditioning information from batch.
        For LightGen, we condition on material properties and geometry.
        """
        # Extract material and geometry info
        albedo_map = batch['albedo_map']  # [B, 3, H, W]
        metal_map = batch['metal_map']  # [B, 1, H, W]
        rough_map = batch['rough_map']  # [B, 1, H, W]
        alpha_map = batch.get('alpha_map', None)  # [B, 1, H, W] or None (no-alpha variants)
        position_map = batch['position_map']  # [B, 3, H, W]
        normal_map = batch['normal_map']  # [B, 3, H, W]
        
        # Pack all material properties into a single tensor for conditioning
        # This gives us: normal(3) + albedo(3) + metal(1) + rough(1) = 8 channels
        material_cond = torch.cat([normal_map, albedo_map, metal_map, rough_map], dim=1)  # [B, 8, H, W]
        
        # Convert to expected format: [B, V, H, W, C]
        B, C, H, W = material_cond.shape
        rgb_cond = material_cond.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, 8]
        
        # Use fixed prompt for LightGen (dataset doesn't have text prompts)
        # Using consistent prompt rather than scene_id hashes which carry no semantic meaning
        prompt = ["emission generation"] * B
        
        # Generate text embeddings (cache since prompt is always "emission generation")
        if not hasattr(self, '_cached_text_embedding'):
            self._cached_text_embedding = self.image_tokenizer.process_text(["emission generation"]).to(dtype=self.dtype)  # [1, 768]
        text_embeddings = self._cached_text_embedding.expand(B, -1)

        # Use precomputed CLIP image embedding if available, otherwise encode online.
        # NOTE: default collate turns per-sample None into [None] (a list), so guard on tensor-ness,
        # not `is not None` — otherwise online-CLIP inference calls `.to()` on a list.
        if isinstance(batch.get('clip_image_embedding'), torch.Tensor):
            image_embeddings = batch['clip_image_embedding'].to(dtype=self.dtype)  # [B, 768]
        elif isinstance(batch.get('thumbnail'), torch.Tensor):
            rendered_thumbnail = batch['thumbnail']  # [B, 1, H, W, 3]
            image_embeddings = self.image_tokenizer.process_image(rendered_thumbnail).to(dtype=self.dtype)
        else:
            # Fallback: use albedo UV map if thumbnail not available
            spuv.warn("Thumbnail not found in batch, using albedo UV map as fallback")
            albedo_for_clip = albedo_map.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, 3]
            image_embeddings = self.image_tokenizer.process_image(albedo_for_clip).to(dtype=self.dtype)
        
        # GT emission mask (binary, where emission > 0) - used as input condition when available
        gt_emission_mask = batch.get('gt_emission_mask', None)  # [B, 1, H, W] or None
        
        condition_info = {
            'rgb_cond': rgb_cond,  # Contains all material properties [B, 1, H, W, 8]
            'text_embeddings': text_embeddings,
            'image_embeddings': image_embeddings,
            'prompt': prompt,
            'albedo_map': albedo_map,
            'metal_map': metal_map,
            'rough_map': rough_map,
            'alpha_map': alpha_map,
            'normal_map': normal_map,
            'gt_emission_mask': gt_emission_mask,
        }
        
        return condition_info
    
    def forward(self, condition: Dict[str, Any], diffusion_data: Dict[str, Any], condition_drop=None) -> Dict[str, Any]:
        """
        Override forward to work directly in UV space without 3D-to-UV baking.
        Since our data is already in UV space, we pass pre-baked material properties directly.
        
        This works with both SimpleUVUNet and PointUVNet.
        """
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        # Get CLIP embeddings (for PointUVNet compatibility)
        text_embeddings = condition.get("text_embeddings", None)
        image_embeddings = condition.get("image_embeddings", None)
        # PointUVNet expects clip_embeddings as a list [text, image]
        if text_embeddings is not None and image_embeddings is not None:
            clip_embeddings = [text_embeddings, image_embeddings]
        else:
            clip_embeddings = None

        # Get material properties (already in UV space)
        albedo_map = condition.get("albedo_map")  # [B, 3, H, W]
        normal_map = condition.get("normal_map")  # [B, 3, H, W]
        metal_map = condition.get("metal_map")    # [B, 1, H, W]
        rough_map = condition.get("rough_map")    # [B, 1, H, W]
        alpha_map = condition.get("alpha_map", None)  # [B, 1, H, W] or None

        # Prepare baked_texture: include all material properties (albedo + metallic + roughness
        # [+ alpha]). Alpha belongs HERE rather than as a later concat element: this is the only
        # placement that puts it before the occupancy mask in the backbone's input order, gives it
        # the same [0,1]->[-1,1] remap as the other material channels, and keeps base_channels
        # equal to in_channels so the gt_emission_mask oracle branch stays off.
        if alpha_map is not None:
            baked_texture = torch.cat([albedo_map, metal_map, rough_map, alpha_map], dim=1)  # [B, 6, H, W]
        else:
            baked_texture = torch.cat([albedo_map, metal_map, rough_map], dim=1)  # [B, 5, H, W]
        baked_weights = mask_map     # [B, 1, H, W]
        
        # GT emission mask condition (if available)
        gt_emission_mask = condition.get("gt_emission_mask", None)
        
        # Prepare image info with conditioning
        image_info = {
            'rgb_cond': condition.get("rgb_cond"),  # Contains all material properties [B, 1, H, W, 8]
            'baked_texture': baked_texture,  # For PointUVNet: pre-baked material in UV space
            'baked_weights': baked_weights,  # For PointUVNet: occupancy mask
            'gt_emission_mask': gt_emission_mask,  # [B, 1, H, W] binary mask or None
        }

        # Get batch size from mask_map if input_tensor is None (eval mode)
        if input_tensor is not None:
            batch_size = input_tensor.shape[0]
            device = input_tensor.device
        else:
            batch_size = mask_map.shape[0]
            device = mask_map.device
        
        if condition_drop is None and self.training:
            condition_drop = torch.rand(batch_size, device=device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(batch_size, device=device)
        
        # Call the backbone (works with both SimpleUVUNet and PointUVNet)
        # For PointUVNet: pass clip_embeddings list
        # For SimpleUVUNet: it expects image_embeddings but doesn't use it, so clip_embeddings works too
        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps,
           clip_embeddings,  # Changed from image_embeddings to support PointUVNet
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info
    
    def prepare_diffusion_data(self, batch, noisy_images=None):
        """
        Prepare diffusion data from batch.
        Uses Flow Matching from parent class (matches original TEXGen).
        Supports two modes:
        - out_channels=3: Predict RGB emission
        - out_channels=1: Predict binary mask only
        """
        device = get_device()
        B = batch['gt_emission'].shape[0]
        
        # Check model output channels
        out_channels = self.cfg.backbone.out_channels
        
        if out_channels == 1:
            # MASK-ONLY MODE: Convert GT emission to binary mask
            gt_emission = batch['gt_emission']  # [B, 3, H, W] in [-1, 1]
            gt_emission_01 = (gt_emission + 1.0) / 2.0  # Convert to [0, 1]
            emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
            
            # Create binary mask: 1 where emission > threshold, -1 where <= threshold (for [-1, 1] space)
            gt_mask_binary = (gt_emission_01.max(dim=1, keepdim=True)[0] > emissive_threshold).float()  # [B, 1, H, W] in [0, 1]
            sample_images = gt_mask_binary * 2.0 - 1.0  # Convert to [-1, 1] for diffusion
        else:
            # RGB EMISSION MODE: Use GT emission directly
            sample_images = batch['gt_emission']  # [B, 3, H, W] in [-1, 1]
        
        # Mask and position
        mask_map = batch['mask_map']  # [B, 1, H, W]
        position_map = batch['position_map']  # [B, 3, H, W]
        
        # Sample timesteps uniformly in [0, 1] with power transformation (like original TEXGen)
        uniform_samples = torch.rand(B, device=device)
        power = 2  # Skew towards smaller t (more noise)
        timesteps = uniform_samples ** power
        
        # Add noise using Flow Matching (inherited from parent)
        if noisy_images is not None:
            noisy_images = noisy_images.to(dtype=self.dtype)
        else:
            noise = torch.randn_like(sample_images, dtype=self.dtype)
            if sample_images is not None:
                noisy_images = self.get_conditional_flow(noise, sample_images, timesteps)
            else:
                noisy_images = noise
        
        noisy_images *= mask_map
        
        loss_weights = torch.ones_like(timesteps, device=device, dtype=self.dtype)
        
        diffusion_data = {
            'sample_images': sample_images,  # Either [B, 3, H, W] RGB or [B, 1, H, W] mask
            'noisy_images': noisy_images,
            'mask_map': mask_map,
            'position_map': position_map,
            'timesteps': timesteps,
            'noise': noise,
            'batch_loss_weights': loss_weights,
        }
        
        return diffusion_data
    
    def training_step(self, batch, batch_idx):
        """Training step for LightGen"""
        if batch is None:
            return None
        
        # Memory tracking: log at start of training step
        if batch_idx == 0 or batch_idx % 10 == 0:
            log_memory(f"train_step_start (epoch={self.current_epoch}, batch={batch_idx})", self.global_step)
        
        # Prepare data
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)
        
        if batch_idx % 10 == 0:
            log_memory(f"after_data_prep (batch={batch_idx})", self.global_step)
        
        # Forward pass
        out, addition_info = self(condition_info, diffusion_data)
        
        if batch_idx % 10 == 0:
            log_memory(f"after_forward (batch={batch_idx})", self.global_step)
        
        # Compute loss
        loss_dict = self.get_diffusion_loss(out, diffusion_data)
        
        if batch_idx % 10 == 0:
            log_memory(f"after_loss_compute (batch={batch_idx})", self.global_step)
        
        # Log losses per-step only (x-axis is trainer/global_step via define_metric)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # Total loss
        total_loss = sum(loss_dict.values())
        self.log('train/loss', total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # Store outputs for visualization
        outputs = {
            'texture_map_outputs': {
                'pred': out,
                'gt': diffusion_data['sample_images'],
            },
            'mask_map': diffusion_data['mask_map'],
            'render_out': None,
            'render_gt': None,
            'rgb_cond': condition_info['rgb_cond'],
        }
        
        # Visualization with wandb logging
        self.on_check_train(batch, outputs)
        
        # Log to wandb at the same frequency as check_train_every_n_steps
        if hasattr(self, '_wandb_logger') and self._wandb_logger is not None:
            n = self.cfg.check_train_every_n_steps
            if n > 0 and self.global_step % n == 0:
                import wandb as _wandb
                if _wandb.run is not None:
                    batch_size = out.shape[0]
                    emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
                    images = []

                    for i in range(batch_size):
                        s = f"S{i}"

                        # Denoised prediction for sample i
                        pred_x0_i = self.get_batched_pred_x0(
                            out[i:i+1],
                            diffusion_data['timesteps'][i:i+1],
                            diffusion_data['noisy_images'][i:i+1]
                        )
                        is_mask_only = (pred_x0_i.shape[1] == 1)

                        mask_i = diffusion_data['mask_map'][i:i+1]
                        pred_img_i = torch.clamp((pred_x0_i * 0.5 + 0.5) * mask_i, 0, 1)
                        gt_img_i   = torch.clamp((diffusion_data['sample_images'][i:i+1] * 0.5 + 0.5) * mask_i, 0, 1)

                        # Albedo condition
                        if 'albedo_map' in batch and batch['albedo_map'] is not None:
                            albedo_vis_i = torch.clamp(batch['albedo_map'][i:i+1] * mask_i, 0, 1)
                            images.append(_wandb.Image(albedo_vis_i[0].cpu().permute(1, 2, 0).detach().numpy().copy(), caption=f"{s} Input Albedo (UV)"))

                        # Thumbnail condition
                        if 'thumbnail' in batch and batch['thumbnail'] is not None:
                            thumb = batch['thumbnail'][i]  # [1, H, W, 3] or [H, W, 3]
                            if thumb.dim() == 4:
                                thumb = thumb[0]
                            images.append(_wandb.Image(thumb.cpu().detach().numpy().copy(), caption=f"{s} Input Rendering"))

                        # GT emission mask condition
                        if 'gt_emission_mask' in batch and batch['gt_emission_mask'] is not None:
                            gt_emask_cond = batch['gt_emission_mask'][i:i+1] * mask_i  # [1, 1, H, W]
                            gt_emask_vis = gt_emask_cond.repeat(1, 3, 1, 1)  # [1, 3, H, W]
                            images.append(_wandb.Image(gt_emask_vis[0].cpu().permute(1, 2, 0).detach().numpy().copy(), caption=f"{s} GT Emission Mask (condition)"))

                        if is_mask_only:
                            pred_vis = pred_img_i.repeat(1, 3, 1, 1)
                            gt_vis   = gt_img_i.repeat(1, 3, 1, 1)
                            pred_bin = (pred_img_i > 0.5).float().repeat(1, 3, 1, 1)
                            gt_bin   = (gt_img_i   > 0.5).float().repeat(1, 3, 1, 1)
                            images.extend([
                                _wandb.Image(pred_vis[0].cpu().permute(1, 2, 0).detach().numpy().copy(), caption=f"{s} Predicted Mask (continuous)"),
                                _wandb.Image(gt_vis[0].cpu().permute(1, 2, 0).detach().numpy().copy(),   caption=f"{s} GT Mask (continuous)"),
                                _wandb.Image(pred_bin[0].cpu().permute(1, 2, 0).detach().numpy().copy(), caption=f"{s} Predicted Mask (binary, >0.5)"),
                                _wandb.Image(gt_bin[0].cpu().permute(1, 2, 0).detach().numpy().copy(),   caption=f"{s} GT Mask (binary)"),
                            ])
                        else:
                            gt_emask_i   = (gt_img_i.max(dim=1, keepdim=True)[0]   > emissive_threshold).float().repeat(1, 3, 1, 1)
                            pred_emask_i = (pred_img_i.max(dim=1, keepdim=True)[0] > emissive_threshold).float().repeat(1, 3, 1, 1)
                            images.extend([
                                _wandb.Image(pred_img_i[0].cpu().permute(1, 2, 0).detach().numpy().copy(),   caption=f"{s} Predicted Emission"),
                                _wandb.Image(gt_img_i[0].cpu().permute(1, 2, 0).detach().numpy().copy(),     caption=f"{s} Ground Truth"),
                                _wandb.Image(gt_emask_i[0].cpu().permute(1, 2, 0).detach().numpy().copy(),   caption=f"{s} GT Emission Mask (>{emissive_threshold})"),
                                _wandb.Image(pred_emask_i[0].cpu().permute(1, 2, 0).detach().numpy().copy(), caption=f"{s} Pred Emission Mask (>{emissive_threshold})"),
                            ])

                    _wandb.log({"train/predictions": images, "trainer/global_step": self.global_step, "epoch": self.current_epoch})
                    del images
        
        # Explicit cleanup of large intermediate tensors after every training step
        del out, addition_info, outputs
        del diffusion_data, condition_info
        
        # Memory tracking: log after cleanup
        if batch_idx % 10 == 0:
            log_memory(f"train_step_end (batch={batch_idx})", self.global_step)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - images are logged via test tab composite views"""
        # Memory tracking at validation start
        if batch_idx == 0:
            log_memory(f"validation_start (epoch={self.current_epoch})", self.global_step, force=True)
        
        # Call parent validation (computes metrics and saves images)
        # The parent's test_step already logs composite images to test/validation/{object_id}
        result = super().validation_step(batch, batch_idx)
        
        # Memory tracking after validation
        if batch_idx % 10 == 0 or batch_idx == 0:
            log_memory(f"after_validation (batch={batch_idx})", self.global_step)
        
        # Explicit cleanup for memory management
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if batch_idx % 10 == 0 or batch_idx == 0:
            log_memory(f"after_val_cleanup (batch={batch_idx})", self.global_step)
        
        return result
    
    def get_diffusion_loss(self, out, diffusion_data):
        """
        Compute diffusion loss with Flow Matching (inherited from parent).
        Target: velocity field v = x0 - noise
        
        Supports EXCLUSIVE supervision modes:
        1. If lambda_emission_mask > 0: Supervise bright/dark regions separately
        2. Otherwise: Standard MSE+L1 over all regions (baseline TEXGen)
        """
        sample_images = diffusion_data['sample_images']
        noise = diffusion_data['noise']
        mask = diffusion_data['mask_map']
        
        # Flow Matching velocity target (same as original TEXGen)
        target = sample_images - noise
        
        loss_dict = {}
        
        # Check if we're using emission mask loss (which requires exclusive supervision)
        lambda_mask = self.cfg.loss.diffusion_loss_dict.get('lambda_emission_mask', 0.0)
        lambda_dark = self.cfg.loss.diffusion_loss_dict.get('lambda_dark_region', 0.0)
        
        if lambda_mask > 0 or lambda_dark > 0:
            # EXCLUSIVE SUPERVISION MODE: Separate bright and dark regions
            # Compute emissive/dark masks
            gt_emission = (sample_images + 1.0) / 2.0  # Convert to [0, 1]
            emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
            
            bright_mask = (gt_emission.max(dim=1, keepdim=True)[0] > emissive_threshold).float() * mask
            dark_mask = (gt_emission.max(dim=1, keepdim=True)[0] <= emissive_threshold).float() * mask
            
            # Base losses ONLY on bright regions
            if bright_mask.sum() > 0:
                bright_mse = F.mse_loss(out * bright_mask, target * bright_mask, reduction='sum') / (bright_mask.sum() + 1e-8)
                bright_l1 = F.l1_loss(out * bright_mask, target * bright_mask, reduction='sum') / (bright_mask.sum() + 1e-8)
                
                if self.cfg.loss.diffusion_loss_dict.get('lambda_mse', 0.0) > 0:
                    loss_dict['mse_bright'] = bright_mse * self.cfg.loss.diffusion_loss_dict['lambda_mse']
                
                if self.cfg.loss.diffusion_loss_dict.get('lambda_l1', 0.0) > 0:
                    loss_dict['l1_bright'] = bright_l1 * self.cfg.loss.diffusion_loss_dict['lambda_l1']
        else:
            # STANDARD MODE: MSE+L1 over all regions (original TEXGen)
            mse_loss = F.mse_loss(out * mask, target * mask, reduction='mean')
            l1_loss = F.l1_loss(out * mask, target * mask, reduction='mean')
            
            if self.cfg.loss.diffusion_loss_dict.get('lambda_mse', 0.0) > 0:
                loss_dict['mse'] = mse_loss * self.cfg.loss.diffusion_loss_dict['lambda_mse']
            
            if self.cfg.loss.diffusion_loss_dict.get('lambda_l1', 0.0) > 0:
                loss_dict['l1'] = l1_loss * self.cfg.loss.diffusion_loss_dict['lambda_l1']
        
        # Dark region loss - constrain non-emissive regions to be pure black (0 in [0,1] or -1 in [-1,1])
        if lambda_dark > 0:
            # Compute dark mask (will reuse from above if already computed in exclusive mode)
            gt_emission = (sample_images + 1.0) / 2.0
            emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
            dark_mask = (gt_emission.max(dim=1, keepdim=True)[0] <= emissive_threshold).float() * mask
            
            # Only compute loss if there are dark regions
            if dark_mask.sum() > 0:
                # In flow matching: v = x0 - noise
                # For dark regions, we want x0 = -1 (pure black in [-1,1])
                # So the velocity target should be: v = -1 - noise
                # This ensures that when denoised: x0 = v + noise = (-1 - noise) + noise = -1
                dark_target = -torch.ones_like(sample_images) - noise
                
                # MSE loss on velocity in dark regions, pushing towards pure black output
                dark_mse = F.mse_loss(
                    out * dark_mask,
                    dark_target * dark_mask,
                    reduction='sum'
                ) / (dark_mask.sum() + 1e-8)
                
                loss_dict['dark_region_mse'] = dark_mse * lambda_dark
                # Log statistics (aggregated per epoch)
                if self.training:
                    dark_ratio = dark_mask.sum() / (mask.sum() + 1e-8)
                    self.log('train/dark_ratio', dark_ratio, on_step=True, on_epoch=False, prog_bar=False)
        
        # Predicted-emission mask classification loss
        # GT mask = uv_mask AND (gt_emission > mask_cls_threshold)
        # Predicted mask = threshold pred_x0 emission at mask_cls_threshold (soft via BCE on continuous value)
        lambda_pred_mask_cls = self.cfg.loss.diffusion_loss_dict.get('lambda_pred_mask_cls', 0.0)
        if lambda_pred_mask_cls > 0:
            mask_cls_threshold = self.cfg.loss.diffusion_loss_dict.get('mask_cls_threshold', 0.01)

            # Reconstruct predicted x0 from velocity: x0 = v + noise
            pred_x0 = out + noise
            pred_emission_01 = (pred_x0 + 1.0) / 2.0  # [B, C, H, W] in [0, 1]

            # Predicted mask: max over channels -> continuous value in [0, 1], zeroed outside UV islands
            if pred_emission_01.shape[1] == 1:
                pred_mask_raw = pred_emission_01 * mask
            else:
                pred_mask_raw = pred_emission_01.max(dim=1, keepdim=True)[0] * mask  # [B, 1, H, W]

            # GT mask: uv_mask AND (gt_emission > threshold)
            gt_emission_01 = (sample_images + 1.0) / 2.0
            if gt_emission_01.shape[1] == 1:
                gt_emission_max = gt_emission_01
            else:
                gt_emission_max = gt_emission_01.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            gt_mask_cls = (gt_emission_max > mask_cls_threshold).float() * mask  # uv_mask AND emission>thresh

            # Convert continuous prediction to logits for BCE: logit(p) = log(p / (1-p))
            pred_clamped = torch.clamp(pred_mask_raw, 1e-7, 1.0 - 1e-7)
            pred_logits = torch.log(pred_clamped / (1.0 - pred_clamped))

            # BCE classification loss, only within valid UV regions
            # Optional class balancing to handle emissive/non-emissive pixel imbalance
            mask_cls_balanced = self.cfg.loss.diffusion_loss_dict.get('mask_cls_balanced', False)
            if mask_cls_balanced:
                # Compute per-batch pos_weight = num_negative / num_positive
                num_pos = (gt_mask_cls * mask).sum()
                num_neg = ((1.0 - gt_mask_cls) * mask).sum()
                pos_weight = torch.clamp(num_neg / (num_pos + 1e-8), min=1.0, max=100.0)
                bce_cls = F.binary_cross_entropy_with_logits(
                    pred_logits * mask,
                    gt_mask_cls,
                    pos_weight=pos_weight,
                    reduction='none',
                )
                if self.training and self.global_step % 100 == 0:
                    self.log('train/mask_cls_pos_weight', pos_weight, on_step=True, on_epoch=False)
            else:
                bce_cls = F.binary_cross_entropy_with_logits(
                    pred_logits * mask,
                    gt_mask_cls,
                    reduction='none',
                )
            mask_cls_loss = (bce_cls * mask).sum() / (mask.sum() + 1e-8)
            loss_dict['pred_mask_cls_bce'] = mask_cls_loss * lambda_pred_mask_cls

            if self.training:
                pred_mask_binary = (torch.sigmoid(pred_logits) > 0.5).float() * mask
                intersection = (pred_mask_binary * gt_mask_cls).sum()
                union = ((pred_mask_binary + gt_mask_cls) > 0).float().sum()
                iou = intersection / (union + 1e-8)
                self.log('train/pred_mask_cls_iou', iou, on_step=True, on_epoch=False, prog_bar=True)
                self.log('train/pred_mask_cls_gt_ratio', gt_mask_cls.sum() / (mask.sum() + 1e-8),
                         on_step=True, on_epoch=False, prog_bar=False)

        # Emission mask prediction loss
        # Supports two modes:
        # 1. out_channels=3: Derive mask from RGB max
        # 2. out_channels=1: Direct mask prediction
        lambda_mask = self.cfg.loss.diffusion_loss_dict.get('lambda_emission_mask', 0.0)
        if lambda_mask > 0:
            # Ground truth emission in normalized space [-1, 1]
            # Convert to [0, 1] for thresholding
            gt_emission = (sample_images + 1.0) / 2.0
            emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
            
            # GT emission mask: binary mask where emission is > threshold
            gt_mask_binary = (gt_emission.max(dim=1, keepdim=True)[0] > emissive_threshold).float()
            
            # Predicted x0 from velocity: x0 = v + noise
            pred_x0 = out + noise
            
            # Check if model outputs 1 channel (mask only) or 3 channels (RGB)
            if out.shape[1] == 1:
                # MODE: Direct mask prediction (1 channel output)
                # Model directly outputs mask in [-1, 1], convert to [0, 1]
                pred_mask_raw = (pred_x0 + 1.0) / 2.0  # [B, 1, H, W]
            else:
                # MODE: RGB emission prediction (3 channels), derive mask
                pred_emission = (pred_x0 + 1.0) / 2.0  # [B, 3, H, W]
                pred_mask_raw = pred_emission.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            
            # Convert probabilities to logits: logit(p) = log(p / (1-p))
            # Clamp to avoid log(0) or division by 0
            pred_mask_clamped = torch.clamp(pred_mask_raw, 1e-7, 1.0 - 1e-7)
            pred_mask_logits = torch.log(pred_mask_clamped / (1.0 - pred_mask_clamped))
            
            # Apply UV mask to only compute on valid regions
            valid_mask = mask
            
            # BCE with logits loss (autocast-safe)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_mask_logits * valid_mask,
                gt_mask_binary * valid_mask,
                reduction='none'
            )
            mask_loss = (bce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            loss_dict['emission_mask_bce'] = mask_loss * lambda_mask
            
            # If RGB mode, also enforce dark regions are zero
            if out.shape[1] == 3:
                pred_emission = (pred_x0 + 1.0) / 2.0
                dark_mask = (1.0 - gt_mask_binary)  # Inverse of emissive mask
                if dark_mask.sum() > 0:
                    dark_rgb_target = torch.zeros_like(pred_emission)
                    dark_rgb_loss = F.mse_loss(
                        pred_emission * dark_mask,
                        dark_rgb_target * dark_mask,
                        reduction='sum'
                    ) / (dark_mask.sum() + 1e-8)
                    
                    loss_dict['dark_rgb_mse'] = dark_rgb_loss * lambda_mask * 0.5
            
            # Log statistics (aggregated per epoch)
            if self.training:
                # IoU metric for mask prediction
                pred_mask_probs = torch.sigmoid(pred_mask_logits)
                pred_mask_binary = (pred_mask_probs > 0.5).float() * valid_mask
                gt_mask_valid = gt_mask_binary * valid_mask
                
                intersection = (pred_mask_binary * gt_mask_valid).sum()
                union = ((pred_mask_binary + gt_mask_valid) > 0).float().sum()
                iou = intersection / (union + 1e-8)
                
                self.log('train/mask_iou', iou, on_step=True, on_epoch=False, prog_bar=True)
                self.log('train/mask_gt_ratio', gt_mask_valid.sum() / (valid_mask.sum() + 1e-8), 
                        on_step=True, on_epoch=False, prog_bar=False)
        
        return loss_dict
    
    def get_batched_pred_x0(self, out, timesteps, noisy_input):
        """
        Get predicted x0 from model output based on prediction type.
        """
        # Ensure alphas_cumprod is on the same device as timesteps
        device = timesteps.device
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        
        # Convert timesteps to long type for indexing
        timesteps = timesteps.long()
        
        if self.prediction_type == "epsilon":
            # out is predicted noise
            alpha_prod_t = alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = (noisy_input - beta_prod_t.sqrt().view(-1, 1, 1, 1) * out) / alpha_prod_t.sqrt().view(-1, 1, 1, 1)
        elif self.prediction_type == "sample":
            # out is directly predicted x0
            pred_x0 = out
        elif self.prediction_type == "v_prediction":
            # out is v-prediction
            alpha_prod_t = alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = alpha_prod_t.sqrt().view(-1, 1, 1, 1) * noisy_input - beta_prod_t.sqrt().view(-1, 1, 1, 1) * out
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return pred_x0
    
    # test_pipeline method removed - using parent's Flow Matching-compatible version
    # Parent class (texgen_test.py) uses simple Euler integration: x += Δt*v
    # This matches the Flow Matching training: x_t = (1-t)*ε + t*x_0
    #
    # The old DDIM-based implementation below did NOT match training:
    # - Used DDIM scheduler with discrete timesteps [0, 999]
    # - Used complex DDIM.step() instead of simple Euler
    # - Training was on continuous t ∈ [0, 1] with linear interpolation
    #
    # To re-enable DDIM (not recommended), uncomment the method below

