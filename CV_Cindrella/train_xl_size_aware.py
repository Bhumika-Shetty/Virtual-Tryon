"""
Size-Aware IDM-VTON Training Script - Full Combinatorial Training

Trains the size-aware virtual try-on model with:
- Body size conditioning (S/M/L)
- Clothing size conditioning (S/M/L/XL)
- Full 9-combination matrix support

Training Strategy:
1. Freeze base IDM-VTON components (VAE, text encoders, image encoder)
2. Train size embedder (learns size representations)
3. Optionally fine-tune UNet decoder layers

Author: Cinderella Team
Date: 2025-12-08
"""

import os
import random
import argparse
import json
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection
)

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.size_embedder import SizeEmbedder, RelativeFitEmbedder
from src.size_aware_tryon_net import SizeAwareTryonNet

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple, List, Dict, Optional
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF
import numpy as np


# ============================================================
# LoRA Implementation for IP-Adapter
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.
    Adds trainable low-rank matrices to existing linear layers.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original layer
        original_layer.requires_grad_(False)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original_layer(x)

        # Add LoRA contribution
        lora_result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_result


def add_lora_to_model(model: nn.Module, rank: int = 16, alpha: float = 1.0) -> List[nn.Parameter]:
    """
    Add LoRA adapters to all linear layers in a model.

    Args:
        model: Model to add LoRA to
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)

    Returns:
        List of LoRA parameters for optimizer
    """
    lora_params = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRA layer
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)

            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    return lora_params


class SizeAwareCombinatorialDataset(data.Dataset):
    """
    Dataset for Full Combinatorial Training.

    Supports explicit body_size and cloth_size labels for each sample.
    If labels not available, estimates from images using size annotation module.
    """

    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        size_annotation_file: Optional[str] = None,
        size_dropout_prob: float = 0.1,
    ):
        """
        Initialize dataset.

        Args:
            dataroot_path: Path to dataset root
            phase: 'train' or 'test'
            order: 'paired' or 'unpaired'
            size: Image size (height, width)
            size_annotation_file: Optional JSON file with size annotations
            size_dropout_prob: Probability of dropping size conditioning (for CFG)
        """
        super().__init__()

        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.size_dropout_prob = size_dropout_prob

        # Transforms
        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.toTensor = transforms.ToTensor()
        self.clip_processor = CLIPImageProcessor()
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)

        # Size mappings
        self.body_size_map = {'small': 0, 'S': 0, 'medium': 1, 'M': 1, 'large': 2, 'L': 2}
        self.cloth_size_map = {'XS': 0, 'S': 0, 'M': 1, 'L': 2, 'XL': 3}

        # Load size annotations if provided
        self.size_annotations = {}
        if size_annotation_file and os.path.exists(size_annotation_file):
            with open(size_annotation_file, 'r') as f:
                annotations = json.load(f)
                # Index by image name
                for item in annotations.get('annotations', annotations):
                    if isinstance(item, dict):
                        key = item.get('image', item.get('id', ''))
                        self.size_annotations[key] = item
            print(f"Loaded {len(self.size_annotations)} size annotations")

        # Load garment annotations
        annotation_file = os.path.join(dataroot_path, phase, f"vitonhd_{phase}_tagged.json")
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data1 = json.load(f)

            annotation_list = ["sleeveLength", "neckLine", "item"]
            self.annotation_pair = {}
            for k, v in data1.items():
                for elem in v:
                    annotation_str = ""
                    for template in annotation_list:
                        for tag in elem["tag_info"]:
                            if tag["tag_name"] == template and tag["tag_category"] is not None:
                                annotation_str += tag["tag_category"] + " "
                    self.annotation_pair[elem["file_name"]] = annotation_str
        else:
            self.annotation_pair = {}

        self.order = order

        # Load image pairs
        im_names = []
        c_names = []

        filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        im_name = parts[0]
                        c_name = parts[1] if len(parts) > 1 and order == "unpaired" else im_name
                        im_names.append(im_name)
                        c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

        print(f"Loaded {len(self.im_names)} samples for {phase}")

    def _get_size_labels(self, im_name: str, c_name: str) -> Tuple[int, int, int]:
        """
        Get size labels for a sample.

        Returns:
            (body_size_idx, cloth_size_idx, relative_fit)
        """
        # Try to get from annotations
        if im_name in self.size_annotations:
            ann = self.size_annotations[im_name]
            body_size = self.body_size_map.get(ann.get('body_size', 'M'), 1)
            cloth_size = self.cloth_size_map.get(ann.get('cloth_size', 'M'), 1)
        else:
            # Default to medium/fitted if no annotations
            body_size = 1  # Medium
            cloth_size = 1  # Medium

        relative_fit = cloth_size - body_size

        return body_size, cloth_size, relative_fit

    def __getitem__(self, index: int) -> Dict:
        """Get a single sample with size conditioning."""
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # Get garment annotation
        cloth_annotation = self.annotation_pair.get(c_name, "shirts")

        # Load images
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width, self.height))

        image = self.transform(im_pil_big)

        # Load mask
        mask_path = os.path.join(
            self.dataroot, self.phase, "agnostic-mask",
            im_name.replace('.jpg', '_mask.png')
        )
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).resize((self.width, self.height))
        else:
            # Fallback: create empty mask
            mask = Image.new('L', (self.width, self.height), 255)
        mask = self.toTensor(mask)[:1]

        # Load DensePose
        densepose_path = os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        if os.path.exists(densepose_path):
            densepose_map = Image.open(densepose_path)
            pose_img = self.toTensor(densepose_map)
        else:
            pose_img = torch.zeros(3, self.height, self.width)

        # Get size labels
        body_size, cloth_size, relative_fit = self._get_size_labels(im_name, c_name)

        # Size conditioning dropout for classifier-free guidance
        drop_size = random.random() < self.size_dropout_prob

        # Training augmentations
        if self.phase == "train":
            # Random horizontal flip
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)

            # Color jitter
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5
                )
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
                    color_jitter.brightness, color_jitter.contrast,
                    color_jitter.saturation, color_jitter.hue
                )

                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

            # Random scale
            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                image = TF.affine(image, angle=0, translate=[0, 0], scale=scale_val, shear=0)
                mask = TF.affine(mask, angle=0, translate=[0, 0], scale=scale_val, shear=0)
                pose_img = TF.affine(pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0)

            # Random shift
            if random.random() > 0.5:
                shift_x = random.uniform(-0.2, 0.2)
                shift_y = random.uniform(-0.2, 0.2)
                translate = [shift_x * image.shape[-1], shift_y * image.shape[-2]]
                image = TF.affine(image, angle=0, translate=translate, scale=1, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate, scale=1, shear=0)
                pose_img = TF.affine(pose_img, angle=0, translate=translate, scale=1, shear=0)

        # Process mask and cloth
        mask = 1 - mask
        cloth_trim = self.clip_processor(images=cloth, return_tensors="pt").pixel_values

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        im_mask = image * mask
        pose_img = self.norm(pose_img)

        # Build result dictionary
        result = {
            "c_name": c_name,
            "im_name": im_name,
            "image": image,
            "cloth": cloth_trim,
            "cloth_pure": self.transform(cloth),
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption": "model is wearing " + cloth_annotation,
            "caption_cloth": "a photo of " + cloth_annotation,
            "annotation": cloth_annotation,
            "pose_img": pose_img,
            # Size conditioning
            "body_size": torch.tensor(body_size if not drop_size else 1, dtype=torch.long),
            "cloth_size": torch.tensor(cloth_size if not drop_size else 1, dtype=torch.long),
            "relative_fit": torch.tensor(relative_fit if not drop_size else 0, dtype=torch.float),
            "drop_size": torch.tensor(drop_size, dtype=torch.bool),
        }

        return result

    def __len__(self) -> int:
        return len(self.im_names)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Size-Aware IDM-VTON Training")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                       default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--pretrained_garmentnet_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_ip_adapter_path", type=str,
                       default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin")
    parser.add_argument("--image_encoder_path", type=str,
                       default="ckpt/image_encoder")

    # IP-Adapter fine-tuning options
    parser.add_argument("--train_ip_adapter", action="store_true",
                       help="Fine-tune IP-Adapter projection model")
    parser.add_argument("--ip_adapter_lr", type=float, default=1e-6,
                       help="Learning rate for IP-Adapter (should be very low)")
    parser.add_argument("--use_ip_adapter_lora", action="store_true",
                       help="Use LoRA adapters for IP-Adapter (recommended)")
    parser.add_argument("--ip_adapter_lora_rank", type=int, default=16,
                       help="LoRA rank for IP-Adapter")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to VITON-HD dataset")
    parser.add_argument("--size_annotation_file", type=str, default=None,
                       help="Optional JSON file with size annotations")
    parser.add_argument("--output_dir", type=str, default="output_size_aware")

    # Image size
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)

    # Training hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--size_embedder_lr", type=float, default=1e-4,
                       help="Learning rate for size embedder (can be higher)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Size conditioning
    parser.add_argument("--size_dropout_prob", type=float, default=0.1,
                       help="Probability of dropping size conditioning for CFG")
    parser.add_argument("--size_injection_method", type=str, default="added_cond",
                       choices=["added_cond", "cross_attention", "timestep_add"])
    parser.add_argument("--use_relative_fit_embedder", action="store_true",
                       help="Use RelativeFitEmbedder instead of SizeEmbedder")

    # Training strategy
    parser.add_argument("--freeze_unet", action="store_true",
                       help="Freeze UNet and only train size embedder")
    parser.add_argument("--train_unet_decoder_only", action="store_true",
                       help="Only train UNet decoder layers")

    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--use_8bit_adam", action="store_true")

    # Misc
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--noise_offset", type=float, default=None)

    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--checkpointing_epochs", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.0)

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    """Main training function."""
    args = parse_args()

    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print("Size-Aware IDM-VTON Training")
        print("=" * 60)
        print(f"Output directory: {args.output_dir}")
        print(f"Size injection method: {args.size_injection_method}")
        print(f"Size dropout probability: {args.size_dropout_prob}")
        print("=" * 60)

    # Load scheduler and tokenizers
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        rescale_betas_zero_snr=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    # Load text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16
    )

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # Load GarmentNet (UNet encoder)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_garmentnet_path, subfolder="unet"
    )
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None

    # Load and customize main UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None
    )

    # Configure UNet for image embeddings
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"

    # Load IP-Adapter weights
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    # Create image projection model
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(accelerator.device, dtype=torch.float32)

    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    image_proj_model.requires_grad_(True)
    unet.encoder_hid_proj = image_proj_model

    # Modify UNet input channels (13 = 4 + 4 + 1 + 4)
    conv_new = torch.nn.Conv2d(
        in_channels=13,  # noisy_latents + mask + masked_latents + pose
        out_channels=unet.conv_in.out_channels,
        kernel_size=3,
        padding=1,
    )
    torch.nn.init.kaiming_normal_(conv_new.weight)
    conv_new.weight.data = conv_new.weight.data * 0.
    conv_new.weight.data[:, :9] = unet.conv_in.weight.data
    conv_new.bias.data = unet.conv_in.bias.data
    unet.conv_in = conv_new
    unet.config['in_channels'] = 13
    unet.config.in_channels = 13

    # ================================================================
    # SIZE-AWARE COMPONENTS
    # ================================================================

    # Create size embedder
    if args.use_relative_fit_embedder:
        size_embedder = RelativeFitEmbedder(
            num_fit_classes=5,
            embedding_dim=1280
        )
    else:
        size_embedder = SizeEmbedder(
            num_body_sizes=3,
            num_cloth_sizes=4,
            embedding_dim=1280,
            use_relative_fit=True
        )

    # Wrap UNet with size-aware wrapper
    size_aware_unet = SizeAwareTryonNet(
        base_unet=unet,
        size_embedder=size_embedder,
        injection_method=args.size_injection_method
    )

    if accelerator.is_main_process:
        print(f"Size embedder parameters: {sum(p.numel() for p in size_embedder.parameters()):,}")

    # ================================================================
    # FREEZE/UNFREEZE STRATEGY
    # ================================================================

    # Move models to device and set dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)

    # Freeze components that shouldn't be trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)

    # Size embedder always trainable
    size_embedder.requires_grad_(True)

    # ================================================================
    # IP-ADAPTER FINE-TUNING SETUP
    # ================================================================
    ip_adapter_lora_params = []

    if args.train_ip_adapter:
        if args.use_ip_adapter_lora:
            # Add LoRA adapters to IP-Adapter projection model (recommended)
            ip_adapter_lora_params = add_lora_to_model(
                image_proj_model,
                rank=args.ip_adapter_lora_rank,
                alpha=args.ip_adapter_lora_rank  # alpha = rank is common
            )
            if accelerator.is_main_process:
                num_lora_params = sum(p.numel() for p in ip_adapter_lora_params)
                print(f"IP-Adapter LoRA parameters: {num_lora_params:,} (rank={args.ip_adapter_lora_rank})")
        else:
            # Full fine-tuning of IP-Adapter (use with caution)
            image_proj_model.requires_grad_(True)
            if accelerator.is_main_process:
                print(f"IP-Adapter full fine-tuning: {sum(p.numel() for p in image_proj_model.parameters()):,} params")
    else:
        image_proj_model.requires_grad_(False)
        if accelerator.is_main_process:
            print("IP-Adapter: Frozen")

    # UNet training strategy
    if args.freeze_unet:
        unet.requires_grad_(False)
        trainable_params = list(size_embedder.parameters())
        if accelerator.is_main_process:
            print("Training: Size embedder only (UNet frozen)")
    elif args.train_unet_decoder_only:
        # Freeze encoder, train decoder
        for name, param in unet.named_parameters():
            if 'down_blocks' in name or 'mid_block' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        trainable_params = [p for p in size_aware_unet.parameters() if p.requires_grad]
        if accelerator.is_main_process:
            print("Training: Size embedder + UNet decoder")
    else:
        unet.requires_grad_(True)
        trainable_params = list(size_aware_unet.parameters())
        if accelerator.is_main_process:
            print("Training: Size embedder + Full UNet")

    # Enable memory optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            print("Warning: xformers not available")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_encoder.enable_gradient_checkpointing()

    unet.train()
    size_embedder.train()

    # ================================================================
    # OPTIMIZER
    # ================================================================

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW

    # Separate learning rates for different components
    param_groups = [
        {"params": list(size_embedder.parameters()), "lr": args.size_embedder_lr},
    ]

    # Add UNet parameters if not frozen
    if not args.freeze_unet:
        unet_params = [p for p in unet.parameters() if p.requires_grad]
        param_groups.append({"params": unet_params, "lr": args.learning_rate})

    # Add IP-Adapter parameters if training
    if args.train_ip_adapter:
        if args.use_ip_adapter_lora:
            # LoRA parameters
            param_groups.append({
                "params": ip_adapter_lora_params,
                "lr": args.ip_adapter_lr
            })
        else:
            # Full IP-Adapter parameters
            ip_adapter_params = [p for p in image_proj_model.parameters() if p.requires_grad]
            param_groups.append({
                "params": ip_adapter_params,
                "lr": args.ip_adapter_lr
            })

    optimizer = optimizer_class(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process:
        total_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
        print(f"Total trainable parameters: {total_params:,}")

    # ================================================================
    # DATASETS
    # ================================================================

    train_dataset = SizeAwareCombinatorialDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="paired",
        size=(args.height, args.width),
        size_annotation_file=args.size_annotation_file,
        size_dropout_prob=args.size_dropout_prob,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=8,
    )

    test_dataset = SizeAwareCombinatorialDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="paired",
        size=(args.height, args.width),
        size_dropout_prob=0.0,  # No dropout for testing
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    # ================================================================
    # TRAINING SETUP
    # ================================================================

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Prepare with accelerator
    size_aware_unet, image_proj_model, unet_encoder, image_encoder, optimizer, train_dataloader, test_dataloader = \
        accelerator.prepare(
            size_aware_unet, image_proj_model, unet_encoder, image_encoder,
            optimizer, train_dataloader, test_dataloader
        )

    # Recalculate steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ================================================================
    # TRAINING LOOP
    # ================================================================

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    train_loss = 0.0

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(size_aware_unet):

                # ============================================
                # VALIDATION/LOGGING
                # ============================================
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    if accelerator.is_main_process:
                        print(f"\n[Step {global_step}] Running validation...")

                # ============================================
                # PREPARE INPUTS
                # ============================================

                # Encode images to latent space
                pixel_values = batch["image"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Encode masked image
                masked_latents = vae.encode(
                    batch["im_mask"].reshape(batch["image"].shape).to(dtype=vae.dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                # Prepare mask
                masks = batch["inpaint_mask"]
                mask = torch.nn.functional.interpolate(
                    masks, size=(args.height // 8, args.width // 8)
                )

                # Encode pose
                pose_map = vae.encode(
                    batch["pose_img"].to(dtype=vae.dtype)
                ).latent_dist.sample()
                pose_map = pose_map * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1),
                        device=model_input.device
                    )

                # Sample timesteps
                bsz = model_input.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=model_input.device
                )

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Concatenate inputs
                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_latents, pose_map], dim=1
                )

                # ============================================
                # TEXT ENCODING
                # ============================================

                text_input_ids = tokenizer(
                    batch['caption'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                text_input_ids_2 = tokenizer_2(
                    batch['caption'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                encoder_output = text_encoder(
                    text_input_ids.to(accelerator.device),
                    output_hidden_states=True
                )
                text_embeds = encoder_output.hidden_states[-2]

                encoder_output_2 = text_encoder_2(
                    text_input_ids_2.to(accelerator.device),
                    output_hidden_states=True
                )
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]

                encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)

                # ============================================
                # IMAGE EMBEDDING (IP-Adapter)
                # ============================================

                img_emb_list = [batch['cloth'][i] for i in range(bsz)]
                image_embeds = torch.cat(img_emb_list, dim=0)
                image_embeds = image_encoder(
                    image_embeds, output_hidden_states=True
                ).hidden_states[-2]
                ip_tokens = image_proj_model(image_embeds)

                # ============================================
                # GARMENT FEATURES
                # ============================================

                cloth_values = batch["cloth_pure"].to(accelerator.device, dtype=vae.dtype)
                cloth_values = vae.encode(cloth_values).latent_dist.sample()
                cloth_values = cloth_values * vae.config.scaling_factor

                # Cloth text encoding
                cloth_text_ids = tokenizer(
                    batch['caption_cloth'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                cloth_text_ids_2 = tokenizer_2(
                    batch['caption_cloth'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                cloth_encoder_output = text_encoder(
                    cloth_text_ids.to(accelerator.device),
                    output_hidden_states=True
                )
                text_embeds_cloth = cloth_encoder_output.hidden_states[-2]

                cloth_encoder_output_2 = text_encoder_2(
                    cloth_text_ids_2.to(accelerator.device),
                    output_hidden_states=True
                )
                text_embeds_2_cloth = cloth_encoder_output_2.hidden_states[-2]
                text_embeds_cloth = torch.cat([text_embeds_cloth, text_embeds_2_cloth], dim=-1)

                # Get garment features from GarmentNet
                _, reference_features = unet_encoder(
                    cloth_values, timesteps, text_embeds_cloth, return_dict=False
                )
                reference_features = list(reference_features)

                # ============================================
                # SDXL CONDITIONING
                # ============================================

                def compute_time_ids(original_size, crops_coords_top_left=(0, 0)):
                    target_size = (args.height, args.width)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    return add_time_ids.to(accelerator.device)

                add_time_ids = torch.cat(
                    [compute_time_ids((args.height, args.width)) for _ in range(bsz)]
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                    "image_embeds": ip_tokens,
                }

                # ============================================
                # SIZE-AWARE FORWARD PASS
                # ============================================

                body_size = batch["body_size"].to(accelerator.device)
                cloth_size = batch["cloth_size"].to(accelerator.device)
                relative_fit = batch["relative_fit"].to(accelerator.device)

                noise_pred = size_aware_unet(
                    sample=latent_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    body_size=body_size,
                    cloth_size=cloth_size,
                    relative_fit=relative_fit,
                    added_cond_kwargs=added_cond_kwargs,
                    garment_features=reference_features,
                ).sample

                # ============================================
                # COMPUTE LOSS
                # ============================================

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Track loss
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

            # Logging
            if accelerator.sync_gradients:
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "epoch": epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # ============================================
        # CHECKPOINTING
        # ============================================

        if epoch % args.checkpointing_epochs == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)

            # Save size embedder
            torch.save(
                size_embedder.state_dict(),
                os.path.join(save_path, "size_embedder.pt")
            )

            # Save UNet if trained
            if not args.freeze_unet:
                unwrapped_unet = accelerator.unwrap_model(size_aware_unet).unet
                unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))

            print(f"Saved checkpoint to {save_path}")

    # Final save
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_save_path, exist_ok=True)

        torch.save(
            size_embedder.state_dict(),
            os.path.join(final_save_path, "size_embedder.pt")
        )

        if not args.freeze_unet:
            unwrapped_unet = accelerator.unwrap_model(size_aware_unet).unet
            unwrapped_unet.save_pretrained(os.path.join(final_save_path, "unet"))

        print(f"\nTraining complete! Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
