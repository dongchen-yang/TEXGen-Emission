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
from spuv.systems.base import BaseLossConfig, BaseSystem
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

from diffusers import (
    DDPMScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    )


@dataclass
class LossConfig(BaseLossConfig):
    diffusion_loss_dict: dict = field(default_factory=dict)
    render_loss_dict: dict = field(default_factory=dict)

    lambda_mse: Any = 0.0
    lambda_l1: Any = 0.0
    lambda_render_lpips: Any = 0.0
    lambda_render_mse: Any = 0.0
    lambda_render_l1: Any = 0.0

    use_min_snr_weight: bool = False
    use_vgg: bool = False
    p_loss_type: str = "lpips"
    lpips_resize: bool = False


class TEXGenDiffusion(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        loss: LossConfig = field(default_factory=LossConfig)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        data_normalization: bool = True
        render_background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        random_background_color: bool = False

        rescale_betas_zero_snr: bool = False
        train_regression: bool = False
        prediction_type: str = "sample"

        use_ema: bool = True
        ema_decay: float = 0.9999
        val_with_ema: bool = True

        test_num_steps: int = 50
        test_save_json: bool = False

        recon_warm_up_steps: int = 0
        test_scheduler_type: str = "ddim"
        test_save_mid_result: bool = False

        # see On the Importance of Noise Scheduling for Diffusion Models
        # http://arxiv.org/abs/2301.10972
        train_image_scaling: float = 1.0
        condition_drop_rate: float = 0.0
        test_cfg_scale: float = 0.0
        guidance_rescale: float = 0.0
        guidance_interval: Tuple[float, float] = (0.0, 1.0)

        # Cond image augmentation
        cond_rgb_perturb: bool = False
        cond_rgb_perturb_scale: Dict[str, Any] = field(default_factory=lambda: {})

    cfg: Config

    def configure(self):
        super().configure()

        self.train_regression = self.cfg.train_regression

        # Model
        self.backbone = spuv.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.use_ema = self.cfg.use_ema
        self.ema_decay = self.cfg.ema_decay
        self.val_with_ema = self.cfg.val_with_ema
        if self.use_ema:
            self.backbone_ema = LitEma(self.backbone, decay=self.ema_decay)
            spuv.info(f"Keeping EMAs of {len(list(self.backbone_ema.buffers()))}.")

        self.log_register = False
        register_bool = getattr(self.cfg.backbone, 'register_bool', False)

        if register_bool:
            self.log_register = True
            spuv.info("Registering activation norms.")

        # Diffusion noise schedules
        self.prediction_type = self.cfg.prediction_type

        # Important re-configuration
        temp_noise_scheduler = DDPMScheduler.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", subfolder="scheduler",
            prediction_type=self.prediction_type,
            rescale_betas_zero_snr=self.cfg.rescale_betas_zero_snr
        )
        betas = temp_noise_scheduler.betas
        # avoid nan during inference
        betas[-1] = 0.9999 if betas[-1] == 1.0 else betas[-1]

        self.betas = betas
        # Important re-configuration

        self.noise_scheduler = DDPMScheduler(
            prediction_type=self.prediction_type,
            trained_betas=self.betas.numpy(),
        )
        self.num_train_timesteps = self.noise_scheduler.num_train_timesteps
        if self.cfg.test_scheduler_type == "unipc":
            self.test_scheduler = UniPCMultistepScheduler(
                prediction_type=self.prediction_type,
                trained_betas=self.betas.numpy(),
            )
        elif self.cfg.test_scheduler_type == "ddim":
            self.test_scheduler = DDIMScheduler(
                prediction_type=self.prediction_type,
                trained_betas=self.betas.numpy(),
                timestep_spacing="trailing",
            )
        elif self.cfg.test_scheduler_type == "dpm_solver":
            self.test_scheduler = DPMSolverMultistepScheduler(
                prediction_type=self.prediction_type,
                trained_betas=self.betas.numpy(),
            )
        else:
            raise ValueError(f"Invalid test scheduler type: {self.cfg.test_scheduler_type}")

        # NvDiffRasterizer
        self.ctx = NVDiffRasterizerContext('cuda', get_device())
        self.render_background_color = torch.tensor(self.cfg.render_background_color, device=get_device())

        # Metrics for validation
        if self.cfg.loss.p_loss_type == "lpips":
            self.lpips_loss_fn = LPIPS()
        elif self.cfg.loss.p_loss_type == "vgg":
            self.lpips_loss_fn = VGGPerceptualLoss().to(self.device)
        else:
            raise ValueError(f"Invalid perceptual loss type: {self.cfg.loss.p_loss_type}")
        self.ssim_metric_fn = SSIM()
        self.psnr_metric_fn = PSNR()

        # Min SNR weights
        if self.cfg.loss.use_min_snr_weight:
            self.snr_weights = compute_snr_from_scheduler(self.num_train_timesteps, self.noise_scheduler)


    def data_augmentation(self, rgb, background_color):
        with torch.cuda.amp.autocast(enabled=False):
            trans = transform.Compose(
                [
                    transform.RandomAffine(
                        degrees=self.cfg.cond_rgb_perturb_scale["rotate"],
                        translate=self.cfg.cond_rgb_perturb_scale["translate"],
                        scale=self.cfg.cond_rgb_perturb_scale["scale"],
                        fill=background_color.cpu().numpy().tolist(),
                    ),
                ]
            )
            data_type = rgb.dtype
            rgb_perturb = trans(rgb.float()).to(data_type)
        return rgb_perturb


    def generate_random_color(self, hue_range=(0, 1), saturation_range=(0.3, 0.7), value_range=(0.3, 0.7)):
        """
        Generate a random color within specified HSV ranges and return it as an RGB tensor.

        Args:
            hue_range (tuple): Range of hue values (0 to 1).
            saturation_range (tuple): Range of saturation values (0 to 1).
            value_range (tuple): Range of value (brightness) values (0 to 1).

        Returns:
            torch.Tensor: RGB color as a tensor with values normalized between 0 and 1.
        """
        # Generate random hue, saturation, and value within the specified ranges
        h = random.uniform(*hue_range)
        s = random.uniform(*saturation_range)
        v = random.uniform(*value_range)

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(h, s, v)

        # Convert to tensor and return
        return torch.tensor(rgb, dtype=torch.float32)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.backbone_ema(self.backbone)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.backbone_ema.store(self.backbone.parameters())
            self.backbone_ema.copy_to(self.backbone)
            if context is not None:
                spuv.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.backbone_ema.restore(self.backbone.parameters())
                if context is not None:
                    spuv.info(f"{context}: Restored training weights")

    def forward(self,
                condition: Dict[str, Any],
                diffusion_data: Dict[str, Any],
                condition_drop=None,
                ) -> Dict[str, Any]:
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        image_embeddings = None
        mesh = condition["mesh"]

        image_info = {
            'mvp_mtx_cond': condition["mvp_mtx_cond"],
            'rgb_cond': condition["rgb_cond"],
        }

        if condition_drop is None and self.training:
            condition_drop = torch.rand(input_tensor.shape[0], device=input_tensor.device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        #spuv.info(f"Condition drop rate: {condition_drop}")
        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps,
           image_embeddings,
           mesh,
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info

    def try_training_step(self, batch, batch_idx):
        pass

    def prepare_diffusion_data(self, batch, noisy_images=None):
        pass

    def prepare_condition_info(self, batch):
        pass

    def get_diffusion_loss(self, out, diffusion_data):
        pass

    def get_batched_pred_x0(self, out, timesteps, noisy_input):
        pass

    def get_render_loss(self, batch_loss_weights, uv_map_pred, uv_map_gt, mesh, mvp_mtx, height, width):
        pass

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

            self.save_image_grid(
                f"it{self.true_global_step}-train.jpg",
                images,
                name="train_step_output",
                step=None,
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

            self.save_image_grid(
                f"it{self.true_global_step}-train-render.jpg",
                images,
                name="train_step_output",
                step=None,
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)
        # Aggressive memory cleanup to prevent OOM during validation
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if (
                self.true_global_step < self.cfg.recon_warm_up_steps
                or self.cfg.train_regression
        ):
            self.train_regression = True
        else:
            self.train_regression = False

        if batch is None:
            spuv.info("Received None batch, skipping.")
            return None

        try:
            # Disable autocast for validation to avoid bfloat16 issues with spconv
            with torch.cuda.amp.autocast(enabled=False):
                if self.use_ema and self.val_with_ema:
                    with self.ema_scope("Validation with ema weights"):
                        texture_map_outputs = self.test_pipeline(batch)
                else:
                    spuv.info("Validation without ema weights")
                    texture_map_outputs = self.test_pipeline(batch)
        except Exception as e:
            import traceback
            spuv.info(f"Error in test pipeline: {e}")
            spuv.info(f"Full traceback:\n{traceback.format_exc()}")
            return None

        # For validation: compute metrics and save UV maps only
        assert len(batch["scene_id"]) == 1
        save_str = batch["scene_id"][0]
        
        # Get predicted and ground truth UV maps
        pred_x0 = texture_map_outputs["pred_x0"]
        gt_x0 = texture_map_outputs["gt_x0"]
        mask_map = texture_map_outputs["mask_map"]
        
        # Denormalize if needed
        if self.cfg.data_normalization:
            pred_img = (pred_x0 * 0.5 + 0.5) * mask_map
            gt_img = (gt_x0 * 0.5 + 0.5) * mask_map
        else:
            pred_img = pred_x0 * mask_map
            gt_img = gt_x0 * mask_map
        
        # Clamp to valid range [0, 1] before computing metrics
        pred_img = torch.clamp(pred_img, 0, 1)
        gt_img = torch.clamp(gt_img, 0, 1)
        
        # Compute MSE and PSNR on UV space
        mse = torch.mean((pred_img - gt_img) ** 2)
        psnr = -10 * torch.log10(mse + 1e-8)
        
        # Log metrics (aggregated per epoch, x-axis is trainer/global_step via define_metric)
        self.log('val/mse', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Check if mask-only mode (1 channel) or RGB mode (3 channels)
        is_mask_only = (pred_img.shape[1] == 1)
        
        # For mask-only mode, repeat to 3 channels for visualization
        if is_mask_only:
            pred_img_vis = pred_img.repeat(1, 3, 1, 1)
            gt_img_vis = gt_img.repeat(1, 3, 1, 1)
        else:
            pred_img_vis = pred_img
            gt_img_vis = gt_img
        
        # Save UV maps (flipped for visualization)
        for key, img in [("pred_x0", pred_img_vis), ("gt_x0", gt_img_vis)]:
            flip_img = torch.flip(img, dims=[2])
            img_format = [{
                "type": "rgb",
                "img": rearrange(flip_img, "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/{save_str}_{key}.png",
                img_format,
                name=f"test/{key}",
                step=None,
            )
        
        # Get input albedo from batch
        albedo_map = batch.get("albedo_map", None)
        if albedo_map is not None:
            # Albedo is already in [0, 1] range, apply mask
            albedo_img = albedo_map * mask_map
        
        # Get emissive threshold from config
        emissive_threshold = self.cfg.loss.diffusion_loss_dict.get('emissive_threshold', 0.001)
        
        if is_mask_only:
            # MASK-ONLY MODE: pred_img is the mask itself (1 channel)
            # Create binary masks for visualization
            gt_emission_mask = (gt_img > 0.5).float().repeat(1, 3, 1, 1)
            pred_emission_mask = (pred_img > 0.5).float().repeat(1, 3, 1, 1)
        else:
            # RGB MODE: Create GT emission mask (where any RGB channel > threshold)
            gt_emission_mask = (gt_img.max(dim=1, keepdim=True)[0] > emissive_threshold).float()
            gt_emission_mask = gt_emission_mask.repeat(1, 3, 1, 1)  # Repeat to 3 channels for visualization
            
            # Create predicted emission mask (where any RGB channel > threshold)
            pred_emission_mask = (pred_img.max(dim=1, keepdim=True)[0] > emissive_threshold).float()
            pred_emission_mask = pred_emission_mask.repeat(1, 3, 1, 1)  # Repeat to 3 channels for visualization
        
        # Save side-by-side comparison image
        flip_pred = torch.flip(pred_img_vis, dims=[2])
        flip_gt = torch.flip(gt_img_vis, dims=[2])
        flip_mask = torch.flip(mask_map.repeat(1, 3, 1, 1), dims=[2])  # Repeat mask to 3 channels
        flip_gt_emission_mask = torch.flip(gt_emission_mask, dims=[2])
        flip_pred_emission_mask = torch.flip(pred_emission_mask, dims=[2])
        
        # Save individual emission masks
        for key, mask_img in [("gt_emission_mask", gt_emission_mask), ("pred_emission_mask", pred_emission_mask)]:
            flip_mask_img = torch.flip(mask_img, dims=[2])
            mask_format = [{
                "type": "rgb",
                "img": rearrange(flip_mask_img, "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            }]
            
            self.save_image_grid(
                f"it{self.true_global_step}-test/{save_str}_{key}.png",
                mask_format,
                name=f"test/{key}",
                step=None,
            )
        
        # Build comparison list
        comparison_images = []
        if albedo_map is not None:
            flip_albedo = torch.flip(albedo_img, dims=[2])
            comparison_images.append(flip_albedo)
        comparison_images.extend([flip_pred, flip_gt, flip_mask, flip_gt_emission_mask, flip_pred_emission_mask])
        
        comparison_img_format = [{
            "type": "rgb",
            "img": rearrange(torch.cat(comparison_images, dim=-1), "B C H W -> (B H) W C"),
            "kwargs": {"data_format": "HWC"},
        }]
        
        self.save_image_grid(
            f"it{self.true_global_step}-test/{save_str}_comparison.png",
            comparison_img_format,
            name="test/comparison",
            step=None,
        )

        if self.cfg.test_save_json:
            save_str = ""
            for scene_id in batch["scene_id"]:
                save_str += str(scene_id) + " "

            self.save_json(
                f"it{self.true_global_step}-test/{self.global_rank}_{batch_idx}.json",
                save_str,
            )

        if self.cfg.test_save_mid_result and len(texture_map_outputs["mid_result"]) != 0:
            for i, mid_result in enumerate(texture_map_outputs["mid_result"]):
                value = mid_result
                if self.cfg.data_normalization:
                    img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"]
                else:
                    img = value * texture_map_outputs["mask_map"]

                background_color = self.render_background_color
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
                # TODO: UV padding
                pad_img = uv_padding(img.squeeze(0), texture_map_outputs['mask_map'].squeeze(0).squeeze(0),
                                     iterations=2)
                render_out = render_batched_meshes(self.ctx, mesh, pad_img, mvp_mtx, height, width, background_color)
                # render_out = render_batched_meshes(self.ctx, mesh, img, mvp_mtx, height, width, background_color)
                
                # Handle dynamic number of views - arrange horizontally if not a perfect square
                B, V, H, W, C = render_out.shape
                if V == 16:
                    # 4x4 grid
                    img_rearranged = rearrange(render_out, "B (V1 V2) H W C -> (B V1 H) (V2 W) C", V1=4)
                else:
                    # Horizontal arrangement for non-square grids
                    img_rearranged = rearrange(render_out, "B V H W C -> (B H) (V W) C")
                
                img_format = [{
                    "type": "rgb",
                    "img": img_rearranged,
                    "kwargs": {"data_format": "HWC"},
                }]

                self.save_image_grid(
                    f"it{self.true_global_step}-test/{self.global_rank}_{batch_idx}/render_{i}.jpg",
                    img_format,
                    name=f"test_step_output_{self.global_rank}_{batch_idx}",
                    step=None,
                )

    def test_pipeline(self, batch):
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)

        B, C, H, W = diffusion_data["mask_map"].shape
        device = get_device()
        test_num_steps = self.cfg.test_num_steps
        self.test_scheduler.set_timesteps(test_num_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        # Use configured out_channels (3 for RGB, 1 for mask-only)
        out_channels = self.cfg.backbone.out_channels
        noise = torch.randn((B, out_channels, H, W), device=device, dtype=self.dtype)
        noisy_images = noise
        #breakpoint()
        # Only accumulate mid_result if saving is enabled (to prevent memory leaks)
        mid_result = [] if self.cfg.test_save_mid_result else None
        for i, t in enumerate(timesteps):
            if self.train_regression:
                t = torch.tensor([self.num_train_timesteps-1], device=device)
                timestep = t.repeat(B)
                diffusion_data["timesteps"] = timestep
                diffusion_data["noisy_images"] = noisy_images * diffusion_data["mask_map"]
                step_out, addition_info = self(condition_info, diffusion_data)
                noisy_images = step_out
                break
            else:
                timestep = t.repeat(B)
                diffusion_data["timesteps"] = timestep
                diffusion_data["noisy_images"] = noisy_images * diffusion_data["mask_map"]
                if self.cfg.train_image_scaling != 1.0:
                    diffusion_data["noisy_images"] = diffusion_data["noisy_images"] / (
                            diffusion_data["noisy_images"].std(dim=[1, 2, 3], keepdim=True)+1e-6)

                if torch.isnan(diffusion_data["noisy_images"]).any():
                    print("Nan in noisy_images")
                    breakpoint()

                cond_step_out, addition_info = self(condition_info, diffusion_data)
                if (
                        self.cfg.test_cfg_scale != 0.0
                        and self.cfg.guidance_interval[0] <= i / len(timesteps) <= self.cfg.guidance_interval[1]
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
                with torch.cuda.amp.autocast(enabled=False):
                    noisy_images = self.test_scheduler.step(step_out, t, diffusion_data["noisy_images"]).prev_sample
                if torch.isnan(noisy_images).any():
                    print("Nan in noisy_images")
                    breakpoint()
                # Only save mid_result if enabled (to prevent memory leaks)
                if mid_result is not None:
                    mid_result.append(
                        {
                            "x0": step_out,
                            "x_t": diffusion_data["noisy_images"],
                            "x_t_1": noisy_images,
                        }
                    )
            #spuv.info("Test step {timestep}")

        pred_x0 = noisy_images
        texture_map_outputs = {
            "pred_x0": pred_x0,
            "baked_texture": addition_info['baked_texture'],
            "gt_x0": diffusion_data["sample_images"],
            "mask_map": diffusion_data["mask_map"],
            "mid_result": mid_result if mid_result is not None else [],
        }

        return texture_map_outputs
