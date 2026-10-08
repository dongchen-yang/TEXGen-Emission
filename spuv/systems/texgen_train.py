from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import spuv
from spuv.systems.texgen_test import TEXGenDiffusion as TEXGenTestSystem


class TEXGenDiffusion(TEXGenTestSystem):
    """
    Training for the emission model: the training step and the losses that upstream's release leaves
    as stubs in texgen_base.py. Conditioning, the forward pass and inference are texgen_test.py's.

    Input: geometry (position), material (albedo, metallic, roughness, alpha)
    Output: emission map in UV space
    """
    
    @dataclass
    class Config(TEXGenTestSystem.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    cfg: Config

    def training_step(self, batch, batch_idx):
        """Training step for LightGen"""
        if batch is None:
            return None
        
        # Prepare data
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)
        
        # Forward pass
        out, addition_info = self(condition_info, diffusion_data)
        
        # Compute loss
        loss_dict = self.get_diffusion_loss(out, diffusion_data)
        
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
        
        return total_loss

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
