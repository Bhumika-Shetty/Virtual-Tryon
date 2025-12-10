"""
Size-Aware IDM-VTON Inference Script

Run inference with explicit size control:
- Specify body size (S/M/L)
- Specify clothing size (S/M/L/XL)
- Model generates garment with appropriate fit

Author: Cinderella Team
Date: 2025-12-08
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection
)
from diffusers import AutoencoderKL, DDPMScheduler

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.size_embedder import SizeEmbedder, RelativeFitEmbedder
from src.size_aware_tryon_net import SizeAwareTryonNet

from ip_adapter.ip_adapter import Resampler
from typing import List, Optional
import numpy as np


# Size mappings
BODY_SIZE_MAP = {'S': 0, 'small': 0, 'M': 1, 'medium': 1, 'L': 2, 'large': 2}
CLOTH_SIZE_MAP = {'XS': 0, 'S': 0, 'M': 1, 'L': 2, 'XL': 3}

FIT_DESCRIPTIONS = {
    -2: "very tight",
    -1: "tight/snug",
    0: "fitted/perfect",
    1: "loose/relaxed",
    2: "oversized/baggy",
    3: "very oversized"
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Size-Aware IDM-VTON Inference")

    # Input/Output
    parser.add_argument("--person_image", type=str, required=True,
                       help="Path to person image")
    parser.add_argument("--garment_image", type=str, required=True,
                       help="Path to garment image")
    parser.add_argument("--output_dir", type=str, default="output_inference",
                       help="Output directory")

    # Size control
    parser.add_argument("--body_size", type=str, default="M",
                       choices=["S", "M", "L", "small", "medium", "large"],
                       help="Body size category")
    parser.add_argument("--cloth_size", type=str, default="M",
                       choices=["XS", "S", "M", "L", "XL"],
                       help="Clothing size category")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                       default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--pretrained_garmentnet_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_ip_adapter_path", type=str,
                       default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin")
    parser.add_argument("--image_encoder_path", type=str,
                       default="ckpt/image_encoder")
    parser.add_argument("--size_embedder_path", type=str, default=None,
                       help="Path to trained size embedder checkpoint")

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--size_guidance_scale", type=float, default=1.5,
                       help="How strongly to follow size conditioning")
    parser.add_argument("--seed", type=int, default=42)

    # Image settings
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_models(args):
    """Load all required models."""
    device = args.device

    print("Loading models...")

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
    ).to(device, dtype=torch.float16)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).to(device, dtype=torch.float16)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_path
    ).to(device, dtype=torch.float16)

    # Load GarmentNet
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_garmentnet_path, subfolder="unet"
    ).to(device, dtype=torch.float16)
    unet_encoder.config.addition_embed_type = None

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16
    ).to(device)

    # Configure for image embeddings
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"

    # Load IP-Adapter
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    # Image projection
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(device, dtype=torch.float32)
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    unet.encoder_hid_proj = image_proj_model

    # Modify input channels
    conv_new = torch.nn.Conv2d(
        in_channels=13,
        out_channels=unet.conv_in.out_channels,
        kernel_size=3,
        padding=1,
    )
    conv_new.weight.data = torch.zeros_like(conv_new.weight.data)
    conv_new.weight.data[:, :9] = unet.conv_in.weight.data
    conv_new.bias.data = unet.conv_in.bias.data
    unet.conv_in = conv_new
    unet.config.in_channels = 13

    # Load size embedder
    size_embedder = SizeEmbedder(
        num_body_sizes=3,
        num_cloth_sizes=4,
        embedding_dim=1280,
        use_relative_fit=True
    ).to(device)

    if args.size_embedder_path and os.path.exists(args.size_embedder_path):
        print(f"Loading trained size embedder from {args.size_embedder_path}")
        size_embedder.load_state_dict(
            torch.load(args.size_embedder_path, map_location=device)
        )

    # Create size-aware wrapper
    size_aware_unet = SizeAwareTryonNet(
        base_unet=unet,
        size_embedder=size_embedder,
        injection_method='added_cond'
    )

    print("Models loaded successfully!")

    return {
        'noise_scheduler': noise_scheduler,
        'tokenizer': tokenizer,
        'tokenizer_2': tokenizer_2,
        'text_encoder': text_encoder,
        'text_encoder_2': text_encoder_2,
        'vae': vae,
        'image_encoder': image_encoder,
        'unet_encoder': unet_encoder,
        'size_aware_unet': size_aware_unet,
        'size_embedder': size_embedder,
        'image_proj_model': image_proj_model,
    }


def preprocess_images(person_path: str, garment_path: str, args):
    """Preprocess input images."""
    # Load images
    person_img = Image.open(person_path).convert('RGB')
    garment_img = Image.open(garment_path).convert('RGB')

    # Resize
    person_img = person_img.resize((args.width, args.height))
    garment_img = garment_img.resize((args.width, args.height))

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    person_tensor = transform(person_img)
    garment_tensor = transform(garment_img)

    # CLIP processor for garment
    clip_processor = CLIPImageProcessor()
    garment_clip = clip_processor(images=garment_img, return_tensors="pt").pixel_values

    return person_tensor, garment_tensor, garment_clip, person_img, garment_img


def run_inference(
    models: dict,
    person_tensor: torch.Tensor,
    garment_tensor: torch.Tensor,
    garment_clip: torch.Tensor,
    body_size: int,
    cloth_size: int,
    args,
):
    """Run size-aware inference."""
    device = args.device

    # Unpack models
    vae = models['vae']
    text_encoder = models['text_encoder']
    text_encoder_2 = models['text_encoder_2']
    tokenizer = models['tokenizer']
    tokenizer_2 = models['tokenizer_2']
    noise_scheduler = models['noise_scheduler']
    image_encoder = models['image_encoder']
    unet_encoder = models['unet_encoder']
    size_aware_unet = models['size_aware_unet']
    image_proj_model = models['image_proj_model']

    # Prepare inputs
    person_tensor = person_tensor.unsqueeze(0).to(device, dtype=torch.float16)
    garment_tensor = garment_tensor.unsqueeze(0).to(device, dtype=torch.float16)
    garment_clip = garment_clip.to(device, dtype=torch.float16)

    # Create simple mask (full upper body)
    mask = torch.ones(1, 1, args.height, args.width).to(device)

    # Create pose placeholder (ideally use DensePose)
    pose_img = torch.zeros(1, 3, args.height, args.width).to(device, dtype=torch.float16)

    # Encode to latent space
    with torch.no_grad():
        # Encode person
        person_latents = vae.encode(person_tensor).latent_dist.sample()
        person_latents = person_latents * vae.config.scaling_factor

        # Encode garment
        garment_latents = vae.encode(garment_tensor).latent_dist.sample()
        garment_latents = garment_latents * vae.config.scaling_factor

        # Encode pose
        pose_latents = vae.encode(pose_img).latent_dist.sample()
        pose_latents = pose_latents * vae.config.scaling_factor

        # Prepare mask
        mask_latent = torch.nn.functional.interpolate(
            mask, size=(args.height // 8, args.width // 8)
        )

        # Masked person
        masked_person = person_tensor * (1 - mask)
        masked_latents = vae.encode(masked_person).latent_dist.sample()
        masked_latents = masked_latents * vae.config.scaling_factor

    # Text encoding
    prompt = "model is wearing a shirt"
    prompt_cloth = "a photo of a shirt"

    text_ids = tokenizer(prompt, padding="max_length", truncation=True,
                        max_length=77, return_tensors="pt").input_ids.to(device)
    text_ids_2 = tokenizer_2(prompt, padding="max_length", truncation=True,
                            max_length=77, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        encoder_output = text_encoder(text_ids, output_hidden_states=True)
        text_embeds = encoder_output.hidden_states[-2]

        encoder_output_2 = text_encoder_2(text_ids_2, output_hidden_states=True)
        pooled_embeds = encoder_output_2[0]
        text_embeds_2 = encoder_output_2.hidden_states[-2]

    encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)

    # Image embedding
    with torch.no_grad():
        image_embeds = image_encoder(garment_clip, output_hidden_states=True).hidden_states[-2]
        ip_tokens = image_proj_model(image_embeds)

    # Garment features
    cloth_text_ids = tokenizer(prompt_cloth, padding="max_length", truncation=True,
                              max_length=77, return_tensors="pt").input_ids.to(device)
    cloth_text_ids_2 = tokenizer_2(prompt_cloth, padding="max_length", truncation=True,
                                   max_length=77, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        cloth_enc = text_encoder(cloth_text_ids, output_hidden_states=True)
        cloth_embeds = cloth_enc.hidden_states[-2]
        cloth_enc_2 = text_encoder_2(cloth_text_ids_2, output_hidden_states=True)
        cloth_embeds_2 = cloth_enc_2.hidden_states[-2]
        cloth_text_embeds = torch.cat([cloth_embeds, cloth_embeds_2], dim=-1)

        timesteps_dummy = torch.tensor([0], device=device)
        _, reference_features = unet_encoder(
            garment_latents, timesteps_dummy, cloth_text_embeds, return_dict=False
        )
        reference_features = list(reference_features)

    # SDXL conditioning
    add_time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]],
                                device=device)
    added_cond_kwargs = {
        "text_embeds": pooled_embeds,
        "time_ids": add_time_ids,
        "image_embeds": ip_tokens,
    }

    # Size conditioning
    body_size_tensor = torch.tensor([body_size], device=device)
    cloth_size_tensor = torch.tensor([cloth_size], device=device)
    relative_fit = cloth_size - body_size

    print(f"\nSize conditioning:")
    print(f"  Body size: {['S', 'M', 'L'][body_size]}")
    print(f"  Cloth size: {['S', 'M', 'L', 'XL'][cloth_size]}")
    print(f"  Expected fit: {FIT_DESCRIPTIONS.get(relative_fit, 'custom')}")

    # Sampling
    noise_scheduler.set_timesteps(args.num_inference_steps)

    # Initialize latents with noise
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        (1, 4, args.height // 8, args.width // 8),
        generator=generator,
        device=device,
        dtype=torch.float16
    )
    latents = latents * noise_scheduler.init_noise_sigma

    # Denoising loop
    for i, t in enumerate(noise_scheduler.timesteps):
        # Prepare input
        latent_input = torch.cat([latents, mask_latent, masked_latents, pose_latents], dim=1)

        # Predict noise with size conditioning
        with torch.no_grad():
            noise_pred = size_aware_unet(
                sample=latent_input.to(torch.float16),
                timestep=t,
                encoder_hidden_states=encoder_hidden_states.to(torch.float16),
                body_size=body_size_tensor,
                cloth_size=cloth_size_tensor,
                added_cond_kwargs=added_cond_kwargs,
                garment_features=reference_features,
            ).sample

        # Step
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents.to(torch.float16)).sample

    # Post-process
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    result = Image.fromarray(image)

    return result


def main():
    """Main inference function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    models = load_models(args)

    # Preprocess images
    person_tensor, garment_tensor, garment_clip, person_img, garment_img = \
        preprocess_images(args.person_image, args.garment_image, args)

    # Get size indices
    body_size = BODY_SIZE_MAP.get(args.body_size, 1)
    cloth_size = CLOTH_SIZE_MAP.get(args.cloth_size, 1)

    # Run inference
    result = run_inference(
        models=models,
        person_tensor=person_tensor,
        garment_tensor=garment_tensor,
        garment_clip=garment_clip,
        body_size=body_size,
        cloth_size=cloth_size,
        args=args,
    )

    # Save result
    relative_fit = cloth_size - body_size
    fit_desc = FIT_DESCRIPTIONS.get(relative_fit, 'custom').replace('/', '_')
    output_name = f"result_body{args.body_size}_cloth{args.cloth_size}_{fit_desc}.png"
    output_path = os.path.join(args.output_dir, output_name)
    result.save(output_path)

    print(f"\nResult saved to: {output_path}")

    # Also save comparison
    comparison = Image.new('RGB', (args.width * 3, args.height))
    comparison.paste(person_img, (0, 0))
    comparison.paste(garment_img, (args.width, 0))
    comparison.paste(result, (args.width * 2, 0))

    comparison_path = os.path.join(args.output_dir, "comparison.png")
    comparison.save(comparison_path)
    print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
