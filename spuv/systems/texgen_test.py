from dataclasses import dataclass, field
import cv2
import math
import torch
import torch.nn.functional as F
from einops import rearrange

import spuv
from spuv.utils.misc import get_device
from spuv.utils.typing import *
from spuv.utils.misc import time_recorder as tr
from spuv.utils.snr_utils import compute_snr_from_scheduler, get_weights_from_timesteps
from spuv.utils.mesh_utils import uv_padding
from spuv.utils.nvdiffrast_utils import *
from spuv.systems.texgen_base import TEXGenDiffusion as TEXGenBaseSystem


class TEXGenDiffusion(TEXGenBaseSystem):
    @dataclass
    class Config(TEXGenBaseSystem.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    def configure(self):
        super().configure()
        self.image_tokenizer = spuv.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.sigma_min=0.000001

    def get_conditional_flow(self, noise, sample, t):
        t = t[:, None, None, None]
        return (1 - (1 - self.sigma_min) * t) * noise + t * sample

    def prepare_diffusion_data(self, batch, noisy_images=None):
        device = get_device()
        # The data is in UV space already: the target is the emission map in [-1, 1] (or, with
        # out_channels 1, the binary emission mask), and mask and position come from the batch
        batch_size = batch["gt_emission"].shape[0]
        if self.cfg.backbone.out_channels == 1:
            gt_emission_01 = (batch["gt_emission"] + 1.0) / 2.0
            emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
            sample_images = (gt_emission_01.max(dim=1, keepdim=True)[0] > emissive_threshold).float() * 2.0 - 1.0
        else:
            sample_images = batch["gt_emission"]
        mask_map = batch["mask_map"]
        position_map = batch["position_map"]

        # timesteps = torch.rand(batch_size, device=device)
        # Sample uniformly
        uniform_samples = torch.rand(batch_size, device=device)
        # Apply power transformation to skew towards smaller t
        power = 2  # >1 to skew towards 0
        timesteps = uniform_samples ** power

        if noisy_images is not None:
            noisy_images = noisy_images.to(dtype=self.dtype)
        else:
            noise = torch.randn_like(sample_images, dtype=self.dtype)
            if sample_images is not None:
                noisy_images = self.get_conditional_flow(
                        noise,
                        sample_images,
                        timesteps
                    )
            else:
                noisy_images = noise

        noisy_images *= mask_map

        loss_weights = torch.ones_like(timesteps, device=device, dtype=self.dtype)

        diffusion_data = {
            "sample_images": sample_images,
            "position_map": position_map,
            "mask_map": mask_map,
            "timesteps": timesteps,
            "noise": noise,
            "noisy_images": noisy_images,
            "batch_loss_weights": loss_weights,
        }

        return diffusion_data

    def forward(self,
                condition: Dict[str, Any],
                diffusion_data: Dict[str, Any],
                condition_drop=None,
                ) -> Dict[str, Any]:
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        text_embeddings = condition["text_embeddings"]
        image_embeddings = condition["image_embeddings"]
        clip_embeddings = [text_embeddings, image_embeddings]

        # No mesh to bake a condition view from: the material maps go in pre-baked in UV space
        image_info = {
            'rgb_cond': condition["rgb_cond"],
            'baked_texture': torch.cat([condition["albedo_map"], condition["metal_map"],
                                        condition["rough_map"], condition["alpha_map"]], dim=1),
            'baked_weights': mask_map,
            'gt_emission_mask': condition["gt_emission_mask"],
        }

        if condition_drop is None and self.training:
            condition_drop = torch.rand(input_tensor.shape[0], device=input_tensor.device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(input_tensor.shape[0], device=input_tensor.device)

        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps,   # t in [0, 1]: our models were trained without upstream's *1000
           clip_embeddings,
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info

    def prepare_condition_info(self, batch):
        # No mesh to render the condition view from: rgb_cond packs the UV material maps (albedo,
        # metallic, roughness: [B, 1, H, W, 5]), and the CLIP image is the object's thumbnail
        rgb_cond = torch.cat([batch["albedo_map"], batch["metal_map"], batch["rough_map"]], dim=1)
        rgb_cond = rgb_cond.permute(0, 2, 3, 1).unsqueeze(1)

        prompt = ["emission generation"] * rgb_cond.shape[0]   # the data has no text prompts

        if not hasattr(self, "_cached_text_embedding"):   # one fixed prompt: encoded once
            self._cached_text_embedding = self.image_tokenizer.process_text(["emission generation"]).to(dtype=self.dtype)
        text_embeddings = self._cached_text_embedding.expand(rgb_cond.shape[0], -1)
        if isinstance(batch.get("clip_image_embedding"), torch.Tensor):   # precomputed
            image_embeddings = batch["clip_image_embedding"].to(dtype=self.dtype)
        elif isinstance(batch.get("thumbnail"), torch.Tensor):
            image_embeddings = self.image_tokenizer.process_image(batch["thumbnail"]).to(dtype=self.dtype)
        else:
            spuv.warn("Thumbnail not found in batch, using albedo UV map as fallback")
            image_embeddings = self.image_tokenizer.process_image(
                batch["albedo_map"].permute(0, 2, 3, 1).unsqueeze(1)).to(dtype=self.dtype)

        condition_info = {
            "rgb_cond": rgb_cond,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "prompt": prompt,
            "albedo_map": batch["albedo_map"],
            "metal_map": batch["metal_map"],
            "rough_map": batch["rough_map"],
            "alpha_map": batch["alpha_map"],
            "gt_emission_mask": batch["gt_emission_mask"],
        }

        return condition_info

    def on_check_train(self, batch, outputs):
        if (
                self.true_global_step < self.cfg.recon_warm_up_steps
                or self.cfg.train_regression
        ):
            self.train_regression = True
        else:
            self.train_regression = False

        if (
                self.global_rank == 0
                and self.cfg.check_train_every_n_steps > 0
                and self.true_global_step % (self.cfg.check_train_every_n_steps*10) == 0
        ):
            images = []
            texture_map_outputs = outputs["texture_map_outputs"]

            for key, value in texture_map_outputs.items():
                if self.cfg.data_normalization:
                    img = (value * 0.5 + 0.5) * outputs["mask_map"]
                else:
                    img = value * outputs["mask_map"]
                img_format = {
                    "type": "rgb",
                    "img": rearrange(img, "B C H W -> (B H) W C"),
                    "kwargs": {"data_format": "HWC"},
                }
                images.append(img_format)

            # Save to disk only, don't log to WandB
            self.save_image_grid(
                f"it{self.true_global_step}-train.jpg",
                images,
            )

        if outputs['render_out'] is not None:
            images = [
                {
                    "type": "rgb",
                    "img": rearrange(outputs['render_out'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(outputs['render_gt'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(outputs['rgb_cond'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                }
            ]

            # Save to disk only, don't log to WandB
            self.save_image_grid(
                f"it{self.true_global_step}-train-render.jpg",
                images,
            )

    @torch.no_grad()
    def on_validation_epoch_start(self):
        """Switch to EMA weights once at the start of validation"""
        if self.use_ema and self.val_with_ema:
            spuv.info("Validation with EMA weights: Switching to EMA weights")
            self.backbone_ema.store(self.backbone.parameters())
            self.backbone_ema.copy_to(self.backbone)
            self._ema_switched = True
        else:
            self._ema_switched = False
    
    def on_validation_epoch_end(self):
        """Restore training weights once at the end of validation"""
        if self._ema_switched:
            spuv.info("Validation with EMA weights: Restoring training weights")
            self.backbone_ema.restore(self.backbone.parameters())
            self._ema_switched = False
    
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)
        # Aggressive memory cleanup to prevent OOM during validation
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch is None:
            spuv.info("Received None batch, skipping.")
            return None
        try:
            with torch.cuda.amp.autocast(enabled=False):
                # EMA weights are already switched at epoch level, just run inference
                texture_map_outputs = self.test_pipeline(batch)
        except Exception as e:
            import traceback
            spuv.info(f"Error in test pipeline: {e}")
            spuv.info(f"Full traceback:\n{traceback.format_exc()}")
            return None

        render_images = {}
        background_color = self.render_background_color 

        # Get batch size (support batched validation)
        batch_size = len(batch["scene_id"])

        # Compute and log validation metrics (batched computation for efficiency)
        pred_x0 = texture_map_outputs["pred_x0"]
        gt_x0 = texture_map_outputs["gt_x0"]
        mask_map = texture_map_outputs["mask_map"]
        
        # Denormalize if needed for metric computation
        if self.cfg.data_normalization:
            pred_img = (pred_x0 * 0.5 + 0.5) * mask_map
            gt_img = (gt_x0 * 0.5 + 0.5) * mask_map
        else:
            pred_img = pred_x0 * mask_map
            gt_img = gt_x0 * mask_map
        
        # Compute MSE and PSNR on UV space (batched)
        mse = torch.mean((pred_img - gt_img) ** 2)
        psnr = -10 * torch.log10(mse + 1e-8)
        
        # Log metrics (aggregated per epoch, x-axis is trainer/global_step via define_metric)
        self.log('val/mse', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Loop through batch to save individual images and create visualizations
        has_thumbnail = 'thumbnail' in batch and batch['thumbnail'] is not None
        has_albedo = 'albedo_map' in batch and batch['albedo_map'] is not None
        
        for b_idx in range(batch_size):
            save_str = batch["scene_id"][b_idx]
            
            # save prediction to png file
            value = texture_map_outputs["pred_x0"][b_idx:b_idx+1]
            if self.cfg.data_normalization:
                img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"][b_idx:b_idx+1]
            else:
                img = value * texture_map_outputs["mask_map"][b_idx:b_idx+1]
            # Important to flip the uv map for possible meshlab loading, for rendering using NvDiffRasterizer, do not flip!
            flip_img = torch.flip(img, dims=[2])

            img_format = [{
                "type": "rgb",
                "img": rearrange(flip_img, "B C H W-> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            }]

            # Save to disk only (not logged to WandB - will be logged in preview below)
            self.save_image_grid(
                f"it{self.true_global_step}-test/{save_str}.png",
                img_format,
            )
            
            # Create a composite visualization for each object
            
            # Prepare prediction
            pred_value = texture_map_outputs["pred_x0"][b_idx:b_idx+1]
            if self.cfg.data_normalization:
                pred_img_vis = (pred_value * 0.5 + 0.5) * texture_map_outputs["mask_map"][b_idx:b_idx+1]
            else:
                pred_img_vis = pred_value * texture_map_outputs["mask_map"][b_idx:b_idx+1]
            pred_flip = torch.flip(pred_img_vis, dims=[2])
            pred_vis = rearrange(pred_flip, "B C H W-> (B H) W C")
            
            # Prepare ground truth
            gt_value = texture_map_outputs["gt_x0"][b_idx:b_idx+1]
            if self.cfg.data_normalization:
                gt_img_vis = (gt_value * 0.5 + 0.5) * texture_map_outputs["mask_map"][b_idx:b_idx+1]
            else:
                gt_img_vis = gt_value * texture_map_outputs["mask_map"][b_idx:b_idx+1]
            gt_flip = torch.flip(gt_img_vis, dims=[2])
            gt_vis = rearrange(gt_flip, "B C H W-> (B H) W C")
            
            # Prepare input condition (thumbnail)
            if has_thumbnail:
                thumbnail = batch['thumbnail'][b_idx:b_idx+1]  # [1, 1, H, W, 3]
                if thumbnail.dim() == 5:
                    thumbnail_img = thumbnail[0, 0]  # [H, W, 3]
                else:
                    thumbnail_img = thumbnail[0]  # [H, W, 3]
                
                # Resize thumbnail to match UV map height for side-by-side display
                H_uv, W_uv = pred_vis.shape[0], pred_vis.shape[1]
                H_thumb, W_thumb = thumbnail_img.shape[0], thumbnail_img.shape[1]
                
                # Pad thumbnail to match UV map height if needed
                if H_thumb < H_uv:
                    pad_top = (H_uv - H_thumb) // 2
                    pad_bottom = H_uv - H_thumb - pad_top
                    thumbnail_img = torch.nn.functional.pad(
                        thumbnail_img.permute(2, 0, 1),  # [3, H, W]
                        (0, 0, pad_top, pad_bottom),
                        mode='constant',
                        value=0
                    ).permute(1, 2, 0)  # [H, W, 3]
                elif H_thumb > H_uv:
                    # Crop if thumbnail is larger
                    start = (H_thumb - H_uv) // 2
                    thumbnail_img = thumbnail_img[start:start+H_uv]
            
            # Prepare input albedo UV map
            if has_albedo:
                albedo = batch['albedo_map'][b_idx:b_idx+1]  # [1, 3, H, W]
                albedo_masked = albedo * texture_map_outputs["mask_map"][b_idx:b_idx+1]
                albedo_flip = torch.flip(albedo_masked, dims=[2])
                albedo_vis = rearrange(albedo_flip, "B C H W-> (B H) W C")
            
            # Create composite image: [Thumbnail | Albedo | Prediction | Ground Truth]
            composite_imgs = []
            if has_thumbnail:
                composite_imgs.append({"type": "rgb", "img": thumbnail_img, "kwargs": {"data_format": "HWC"}})
            if has_albedo:
                composite_imgs.append({"type": "rgb", "img": albedo_vis, "kwargs": {"data_format": "HWC"}})
            composite_imgs.append({"type": "rgb", "img": pred_vis, "kwargs": {"data_format": "HWC"}})
            composite_imgs.append({"type": "rgb", "img": gt_vis, "kwargs": {"data_format": "HWC"}})
            
            # Log composite image with object identifier
            object_id = save_str
            self.save_image_grid(
                f"it{self.true_global_step}-test/preview/composite_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                composite_imgs,
                name=f"test/validation/{object_id}",
            )
            
            # Also save individual images to disk (but not to WandB) for reference
            for key, suffix in [("pred_x0", "prediction"), ("gt_x0", "ground_truth")]:
                value = texture_map_outputs[key][b_idx:b_idx+1]
                if self.cfg.data_normalization:
                    img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"][b_idx:b_idx+1]
                else:
                    img = value * texture_map_outputs["mask_map"][b_idx:b_idx+1]
                flip_img = torch.flip(img, dims=[2])
                img_format = [{
                    "type": "rgb",
                    "img": rearrange(flip_img, "B C H W-> (B H) W C"),
                    "kwargs": {"data_format": "HWC"},
                }]
                self.save_image_grid(
                    f"it{self.true_global_step}-test/preview/{suffix}_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                    img_format,
                )
            
            if has_thumbnail:
                self.save_image_grid(
                    f"it{self.true_global_step}-test/preview/thumbnail_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                    [{"type": "rgb", "img": thumbnail_img, "kwargs": {"data_format": "HWC"}}],
                )
            
            if has_albedo:
                self.save_image_grid(
                    f"it{self.true_global_step}-test/preview/albedo_{self.global_rank}_{batch_idx}_{b_idx}.jpg",
                    [{"type": "rgb", "img": albedo_vis, "kwargs": {"data_format": "HWC"}}],
                )
        
        # Explicit cleanup of large tensors to prevent memory leaks
        del texture_map_outputs
        del pred_x0, gt_x0, mask_map, pred_img, gt_img
        
        # 3D rendering disabled - only save UV maps for faster validation
        # Uncomment below if you need 3D rendered views
        """
            img = rearrange(img, "B C H W -> B H W C")
            mvp_mtx = batch['mvp_mtx']
            mesh = batch['mesh']
            # Extract integer values from height/width tensors/lists
            if torch.is_tensor(batch['height']):
                height = batch['height'].item()
            elif isinstance(batch['height'], (list, tuple)):
                height = int(batch['height'][0])
            else:
                height = int(batch['height'])
            
            if torch.is_tensor(batch['width']):
                width = batch['width'].item()
            elif isinstance(batch['width'], (list, tuple)):
                width = int(batch['width'][0])
            else:
                width = int(batch['width'])

            pad_img = uv_padding(img.squeeze(0), texture_map_outputs['mask_map'].squeeze(0).squeeze(0), iterations=2)
            
            render_out = render_batched_meshes(self.ctx, mesh, pad_img, mvp_mtx, height, width, background_color)

            # Dynamic view grid layout based on actual number of views
            num_views = render_out.shape[1]
            # Try to make a square-ish grid, prefer more columns than rows
            V1 = int(math.sqrt(num_views))
            V2 = (num_views + V1 - 1) // V1  # Ceiling division
            
            # If not evenly divisible, pad with zeros
            if V1 * V2 != num_views:
                padding = V1 * V2 - num_views
                render_out = torch.cat([render_out, torch.zeros_like(render_out[:, :1]).expand(-1, padding, -1, -1, -1)], dim=1)
            
            img_format = [{
                "type": "rgb",
                "img": rearrange(render_out, "B (V1 V2) H W C -> (B V1 H) (V2 W) C", V1=V1),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/preview/render_{key}_{self.global_rank}_{batch_idx}.jpg",
                img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=None,
            )

            render_images[key] = torch.clamp(rearrange(render_out, "B V H W C -> (B V) C H W"), min=0, max=1)
            """
        
    def test_pipeline(self, batch):
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)

        device = get_device()
        test_num_steps = self.cfg.test_num_steps

        B, C, H, W = diffusion_data["mask_map"].shape
        # Use configured out_channels (3 for RGB, 1 for mask-only)
        out_channels = self.cfg.backbone.out_channels
        noise = torch.randn((B, out_channels, H, W), device=device, dtype=self.dtype)
        noisy_images = noise

        t_span=torch.linspace(0, 1, test_num_steps, device=device, dtype=self.dtype)
        delta = 1.0 / test_num_steps

        for i, t in enumerate(t_span):
            timestep = t.repeat(B)
            diffusion_data["timesteps"] = timestep
            diffusion_data["noisy_images"] = noisy_images
            cond_step_out, addition_info = self(condition_info, diffusion_data)

            if (
                    self.cfg.test_cfg_scale != 0.0
                    and self.cfg.guidance_interval[0] <= t <= self.cfg.guidance_interval[1]
            ):
                uncond_step_out, _ = self(condition_info, diffusion_data, condition_drop=torch.ones(B, device=device))
                step_out = uncond_step_out + self.cfg.test_cfg_scale * (cond_step_out - uncond_step_out)
                # Apply guidance rescale. From paper [Common Diffusion Noise Schedules
                # and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) section 3.4.
                if self.cfg.guidance_rescale != 0:
                    std_pos = cond_step_out.std(dim=list(range(1, cond_step_out.ndim)), keepdim=True)
                    std_cfg = step_out.std(dim=list(range(1, step_out.ndim)), keepdim=True)
                    # Fuse equation 15,16 for more efficient computation.
                    step_out *= self.cfg.guidance_rescale * (std_pos / std_cfg) + (1 - self.cfg.guidance_rescale)
            else:
                step_out = cond_step_out

            noisy_images = noisy_images + delta * step_out

        pred_x0 = noisy_images
        texture_map_outputs = {
            "pred_x0": pred_x0,
            "baked_texture": addition_info['baked_texture'],
            "gt_x0": diffusion_data["sample_images"],
            "mask_map": diffusion_data["mask_map"],
        }

        return texture_map_outputs
