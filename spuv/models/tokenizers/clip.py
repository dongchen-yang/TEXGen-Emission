from dataclasses import dataclass
import random

import torch
import torch.nn as nn
from einops import rearrange
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from diffusers import AutoencoderKL
from transformers import CLIPTextModel

from ...utils.base import BaseModule
from ...utils.typing import *


class ClipTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "lambda/sd-image-variations-diffusers"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.weight_dtype = torch.bfloat16  # TODO hard coding
        pretrained_model_name_or_path = self.cfg.pretrained_model_name_or_path

        from transformers import CLIPTextModel, CLIPTokenizer
        # CLIP-L text encoder and tokenizer: SDXL's copy holds the same weights as SD 3.5's
        text_tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                       subfolder="tokenizer")
        self.register_non_module(
            "text_tokenizer",
            text_tokenizer,
        )

        self.register_non_module(
            "text_encoder",
            CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         subfolder="text_encoder").to(
                self.device, dtype=self.weight_dtype
            ),
        )

        text_encoder = self.non_module("text_encoder")
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        text_encoder.eval()

        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_encoder").to(self.device, dtype=self.weight_dtype)

        feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="feature_extractor")

        self.clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:, None, None].to(
            self.device, dtype=self.weight_dtype)
        self.clip_image_std = torch.as_tensor(feature_extractor.image_std)[:, None, None].to(
            self.device, dtype=self.weight_dtype)

        self.register_non_module(
            "image_encoder",
            image_encoder,
        )

        image_encoder = self.non_module("image_encoder")
        for p in image_encoder.parameters():
            p.requires_grad_(False)
        image_encoder.eval()

        self.register_non_module(
            "feature_extractor",
            feature_extractor,
        )

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]

        image = rearrange(image, "B N H W C -> (B N) C H W")

        imgs_in_proc = (image - self.clip_image_mean) / self.clip_image_std
        imgs_in_proc = TF.resize(imgs_in_proc,
                                 (self.non_module('feature_extractor').crop_size['height'], self.non_module('feature_extractor').crop_size['width']),
                                 interpolation=InterpolationMode.BICUBIC)

        image_encoder = self.non_module("image_encoder")

        with torch.no_grad():  # Ensure no gradients are tracked for memory efficiency
            image_embeddings = image_encoder(imgs_in_proc.to(self.weight_dtype)).image_embeds
        image_embeddings = rearrange(image_embeddings, "(B N) C -> B (N C)", B=batch_size)
        
        # Clean up intermediate tensors
        del imgs_in_proc

        return image_embeddings

    def process_text(self, prompts) -> torch.Tensor:
        text_tokenizer = self.non_module("text_tokenizer")

        prompt_ids = text_tokenizer(
            prompts, max_length=text_tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        text_encoder = self.non_module("text_encoder")

        # local [B, 77, 768] - CLIP-L has 768-dim embeddings
        # text_embeddings = text_encoder(prompt_ids)[0].to(self.device, dtype=self.weight_dtype)
        # global [B, 768] - pooled text embedding
        text_embeddings = text_encoder(prompt_ids)[1].to(self.device, dtype=self.weight_dtype)

        return text_embeddings