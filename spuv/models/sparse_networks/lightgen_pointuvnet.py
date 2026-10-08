"""
Modified PointUVNet for LightGen - works with pre-baked UV data without rasterization.

Key differences from original PointUVNet:
1. Skips the bake_image_feature_to_uv step (no mesh rasterization needed)
2. Uses pre-computed position_map and material properties from UV data
3. Otherwise identical architecture for 3D-aware feature processing
"""

import torch
import torch.nn as nn
from einops import rearrange

import spuv
from .texgen_network import (
    PointUVNet, 
    PointUVStage, 
    ConditionEmbedding,
    downsample_feature_with_mask,
    upsample_feature_with_mask,
)
from spuv.utils.misc import time_recorder as tr


class LightGenPointUVNet(PointUVNet):
    """
    PointUVNet modified for pre-baked UV data - no rasterization needed.
    
    Accepts material properties directly in UV space instead of baking from 3D views.
    This is perfect for LightGen where all data is already in UV space.
    """
    
    def forward(self,
                x_dense,
                mask_map,
                position_map,
                timestep,
                clip_embeddings,
                image_info,
                data_normalization,
                condition_drop,
                ):
        """
        Forward pass with pre-baked UV data.
        
        Args:
            x_dense: [B, 3, H, W] - noisy emission map
            mask_map: [B, 1, H, W] - occupancy mask
            position_map: [B, 3, H, W] - 3D positions in UV space
            timestep: [B] - diffusion timestep
            clip_embeddings: [text_emb, image_emb] - CLIP embeddings (can be None)
            image_info: dict with 'baked_texture' [B, 5 or 6, H, W] (albedo+metal+rough[+alpha])
                        and 'baked_weights' [B, 1, H, W] (occupancy mask)
            data_normalization: bool - whether data is normalized to [-1, 1]
            condition_drop: [B] - dropout mask for classifier-free guidance
        """
        skip_x = x_dense
        
        # Get pre-baked material properties from image_info
        # These are already in UV space from the dataloader
        if 'baked_texture' in image_info and 'baked_weights' in image_info:
            # Use pre-baked data directly (no rasterization needed!)
            baked_texture = image_info['baked_texture']  # [B, C, H, W] - material properties
            baked_weights = image_info['baked_weights']  # [B, 1, H, W] - mask/weights
            
            # Ensure proper normalization
            #[TODO] ensure normalization is correct
            if data_normalization and baked_texture.max() > 1.0:
                baked_texture = (baked_texture / 255.0) * 2.0 - 1.0
            elif data_normalization and baked_texture.max() <= 1.0 and baked_texture.min() >= 0.0:
                baked_texture = baked_texture * 2.0 - 1.0
        else:
            # Fallback: create dummy baked data if not provided
            # This allows the model to still work even without pre-baked materials
            print("Warning: No pre-baked texture provided, using zeros")
            baked_texture = torch.zeros_like(x_dense)
            baked_weights = mask_map

        # Classifier-free guidance: apply dropout to embeddings
        input_embeddings = []
        if clip_embeddings is not None and len(clip_embeddings) > 0:
            condition_drop_expanded = condition_drop.unsqueeze(-1)
            for i, _ in enumerate(clip_embeddings):
                if clip_embeddings[i] is not None:
                    clip_embedding_null = torch.zeros_like(
                        clip_embeddings[i], device=x_dense.device, dtype=x_dense.dtype
                    )
                    clip_embedding = (
                        condition_drop_expanded * clip_embedding_null + 
                        (1 - condition_drop_expanded) * clip_embeddings[i]
                    )
                    input_embeddings.append(clip_embedding)
                else:
                    # If specific embedding is None, create appropriate size dummy
                    if i == 0:
                        # Text embedding: 1024-dim
                        input_embeddings.append(
                            torch.zeros(x_dense.shape[0], 1024, device=x_dense.device, dtype=x_dense.dtype)
                        )
                    else:
                        # Image embedding: 768-dim
                        input_embeddings.append(
                            torch.zeros(x_dense.shape[0], 768, device=x_dense.device, dtype=x_dense.dtype)
                        )
        else:
            # For LightGen, if no CLIP embeddings provided, create dummy ones
            # Text embedding: 1024-dim, Image embedding: 768-dim
            input_embeddings = [
                torch.zeros(x_dense.shape[0], 1024, device=x_dense.device, dtype=x_dense.dtype),
                torch.zeros(x_dense.shape[0], 768, device=x_dense.device, dtype=x_dense.dtype)
            ]

        # Build input list: [noisy_emission, position, material, mask, (optional) gt_emission_mask]
        # x_dense: [B, 3, H, W] - noisy emission
        # position_map: [B, 3, H, W] - 3D positions
        # baked_texture: [B, 5 or 6, H, W] - albedo(3) + metallic(1) + roughness(1) [+ alpha(1)]
        # baked_weights: [B, 1, H, W] - occupancy mask
        # gt_emission_mask: [B, 1, H, W] - binary GT emission mask (optional, requires in_channels=13)
        concat_list = [x_dense, position_map, baked_texture, baked_weights]
        gt_emission_mask = image_info.get('gt_emission_mask', None)
        if self.cfg.use_gt_emission_mask_cond:
            assert gt_emission_mask is not None, \
                "use_gt_emission_mask_cond=true but the batch carries no gt_emission_mask"
            concat_list.append(gt_emission_mask)
        x_concat = torch.cat(concat_list, dim=1)
        assert x_concat.shape[1] == self.cfg.in_channels, (
            f"assembled input is {x_concat.shape[1]}ch but cfg.in_channels={self.cfg.in_channels}; "
            f"parts={[t.shape[1] for t in concat_list]}"
        )
        x_dense = self.input_conv(x_concat) * mask_map
        
        if torch.isnan(x_dense).any():
            print("x_dense has NaN values after input_conv")
            raise ValueError("NaN detected in input processing")

        # Store pyramid features for skip connections
        pyramid_features = []
        pyramid_mask = []
        pyramid_position = []

        # Generate condition embeddings from timestep and CLIP
        condition_embedding = self.condition_embedder(timestep, input_embeddings)

        # Encoder (downsampling path)
        for scale in range(len(self.block_out_channels)):
            tr.start(f"down{scale}")
            x_dense = getattr(self, f"down{scale}")(
                x_dense,
                mask_map,
                position_map,
                condition_embedding,
                ctx=self.ctx,
                feature_info=None,
            )

            if scale < len(self.block_out_channels) - 1:
                # Store features for skip connections
                pyramid_features.append(x_dense)
                pyramid_mask.append(mask_map)
                pyramid_position.append(position_map)

                # Downsample features and masks
                feature_list, mask_map = downsample_feature_with_mask(
                    [x_dense, position_map], mask_map
                )
                x_dense, position_map = feature_list

                # Project to next scale channels
                x_dense = getattr(self, f"post_conv_down{scale}")(x_dense)
            tr.end(f"down{scale}")

        # Decoder (upsampling path)
        for scale in reversed(range(len(self.block_out_channels) - 1)):
            if scale < len(self.block_out_channels) - 1:
                # Project back to current scale channels
                x_dense = getattr(self, f"pre_conv_up{scale}")(x_dense)

                # Upsample features
                x_dense, _ = upsample_feature_with_mask(x_dense, mask_map)
                mask_map = pyramid_mask[scale]
                position_map = pyramid_position[scale]

                # Skip connection: concatenate with encoder features
                x_dense = torch.cat([x_dense, pyramid_features[scale]], dim=1)
                x_dense = getattr(self, f"skip_conv{scale}")(x_dense)

                # Layer normalization
                B, C, H, W = x_dense.shape
                x_dense = rearrange(x_dense, "B C H W -> (B H W) C")
                x_dense = getattr(self, f"skip_layer_norm{scale}")(x_dense)
                x_dense = rearrange(x_dense, "(B H W) C -> B C H W", B=B, H=H)

            # Process at current scale
            x_dense = getattr(self, f"up{scale}")(
                x_dense,
                mask_map,
                position_map,
                condition_embedding,
                ctx=self.ctx,
                feature_info=None,
            )

        # Final output projection
        x_output = self.output_conv(x_dense)

        # Prepare additional info for output
        addition_info = {
            "pyramid_features": pyramid_features,
            "pyramid_mask": pyramid_mask,
            "pyramid_position": pyramid_position,
            "baked_texture": baked_texture,
            "baked_weights": baked_weights,
        }

        # Apply skip connections (residual from input)
        if self.cfg.skip_input:
            if self.cfg.skip_type == "baked_texture":
                # Skip connection with baked texture (material-based residual)
                # Only use albedo channels (first 3) for skip to match output channels
                baked_albedo = baked_texture[:, :3, :, :]  # [B, 3, H, W]
                return x_output + baked_weights * baked_albedo, addition_info
            elif self.cfg.skip_type == "noise_input":
                # Skip connection with noisy input
                return x_output + skip_x, addition_info
            elif self.cfg.skip_type == "adaptive":
                # Adaptive skip connection with learned gating
                skip_scale = self.ada_skip_scale(condition_embedding)
                x0_scale, input_scale = skip_scale.chunk(2, dim=1)
                x0_scale = x0_scale.unsqueeze(-1).unsqueeze(-1)
                input_scale = input_scale.unsqueeze(-1).unsqueeze(-1)
                
                skip_map = self.ada_skip_map(torch.cat([x_concat, x_dense], dim=1))
                output_scale_map, skip_scale_map = skip_map.chunk(2, dim=1)

                x1 = (1 - output_scale_map) * x_output
                # Only use albedo channels (first 3) for skip to match output channels
                baked_albedo = baked_texture[:, :3, :, :]  # [B, 3, H, W]
                x2 = skip_scale_map * (x0_scale * baked_albedo + input_scale * skip_x)
                x_output = x1 + x2

                addition_info["skip_scale_map"] = skip_scale_map
                addition_info["output_scale_map"] = output_scale_map
                addition_info["skip_scale_rgb"] = x2
                addition_info["output_scale_input"] = x1

                return x_output, addition_info
        else:
            return x_output, addition_info

