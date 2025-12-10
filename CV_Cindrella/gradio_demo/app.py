import sys
import os
import socket
# Add parent directory to path to find src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import cv2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ============================================================
# SIZE-AWARE FUNCTIONS
# ============================================================

def calculate_body_size(densepose_img):
    """
    Calculate the person's body size (S/M/L) based on DensePose body analysis.

    Uses MULTIPLE methods for robust detection:
    1. Overall body silhouette width (all non-black pixels)
    2. Body width as RATIO of image width (resolution-independent)
    3. Torso area analysis

    Args:
        densepose_img: DensePose visualization image (numpy array)

    Returns:
        tuple: (size_label, body_width, recommendation_text)
    """
    img_height, img_width = densepose_img.shape[:2]

    # METHOD 1: Get ALL body pixels (any non-black pixel in DensePose)
    # DensePose colors the entire body, black is background
    gray = cv2.cvtColor(densepose_img, cv2.COLOR_RGB2GRAY) if len(densepose_img.shape) == 3 else densepose_img
    body_mask = gray > 10  # Any non-black pixel is body

    body_coords = np.where(body_mask)

    if len(body_coords[1]) > 0:
        body_left = body_coords[1].min()
        body_right = body_coords[1].max()
        body_width = body_right - body_left

        # METHOD 2: Calculate width RATIO (resolution independent)
        width_ratio = body_width / img_width

        # METHOD 3: Calculate body area ratio
        body_pixels = np.sum(body_mask)
        total_pixels = img_height * img_width
        area_ratio = body_pixels / total_pixels
    else:
        body_width = int(img_width * 0.4)
        width_ratio = 0.4
        area_ratio = 0.15

    # Debug output
    print(f"  [Body Detection] Image size: {img_width}x{img_height}")
    print(f"  [Body Detection] Body width: {body_width}px")
    print(f"  [Body Detection] Width ratio: {width_ratio:.3f} ({width_ratio*100:.1f}% of image)")
    print(f"  [Body Detection] Area ratio: {area_ratio:.3f} ({area_ratio*100:.1f}% of image)")

    # Classification using WIDTH RATIO (works across different image sizes)
    # Small body: < 45% of image width
    # Medium body: 45-55% of image width
    # Large body: > 55% of image width
    #
    # Also use AREA RATIO as secondary check
    # Small: area < 18%
    # Medium: area 18-25%
    # Large: area > 25%

    if width_ratio < 0.45 and area_ratio < 0.20:
        size = "S"
        desc = f"Small frame (width: {width_ratio*100:.0f}%, area: {area_ratio*100:.0f}%)"
    elif width_ratio < 0.55 and area_ratio < 0.28:
        size = "M"
        desc = f"Medium frame (width: {width_ratio*100:.0f}%, area: {area_ratio*100:.0f}%)"
    else:
        size = "L"
        desc = f"Large frame (width: {width_ratio*100:.0f}%, area: {area_ratio*100:.0f}%)"

    print(f"  [Body Detection] Classification: {size} - {desc}")

    return size, body_width, desc


def get_size_recommendation(body_size, selected_size):
    """
    Get size effect based on body size vs selected size.

    BODY-SIZE AWARE PIPELINES:

    === SMALL BODY PIPELINE ===
    - S garment = Perfect fit (1.0)
    - M garment = Loose (1.30x dilation)
    - L garment = Very Oversized (2.5x dilation)

    === MEDIUM BODY PIPELINE ===
    - S garment = Tight (0.70x erosion)
    - M garment = Perfect fit (1.0)
    - L garment = Loose (1.30x dilation)

    === LARGE BODY PIPELINE ===
    - S garment = Very Tight, doesn't fit (0.40x aggressive erosion + crop)
    - M garment = Tight, still doesn't fit well (0.60x erosion)
    - L garment = Perfect fit (1.0)

    Returns:
        tuple: (message, scale_factor, body_pipeline)
    """
    size_order = {"S": 0, "M": 1, "L": 2}
    body_idx = size_order[body_size]
    selected_idx = size_order[selected_size]
    diff = selected_idx - body_idx  # positive = larger garment, negative = smaller

    # ============================================================
    # SMALL BODY PIPELINE (body_size == "S")
    # ============================================================
    if body_size == "S":
        if diff == 0:
            return f"Perfect fit! Size {selected_size} matches your {body_size} body.", 1.0, "small_body"
        elif diff == 1:
            return f"Size {selected_size} will be loose on your {body_size} frame.", 1.30, "small_body"
        elif diff == 2:
            return f"Size {selected_size} will be oversized on your {body_size} frame.", 2.5, "small_body"
        else:
            # This shouldn't happen for small body
            return f"Size {selected_size} on your {body_size} frame.", 1.0, "small_body"

    # ============================================================
    # MEDIUM BODY PIPELINE (body_size == "M")
    # ============================================================
    elif body_size == "M":
        if diff == 0:
            return f"Perfect fit! Size {selected_size} matches your {body_size} body.", 1.0, "medium_body"
        elif diff == 1:
            return f"Size {selected_size} will be loose on your {body_size} frame.", 1.30, "medium_body"
        elif diff == -1:
            return f"Size {selected_size} will be snug on your {body_size} frame.", 0.70, "medium_body"
        else:
            return f"Size {selected_size} on your {body_size} frame.", 1.0, "medium_body"

    # ============================================================
    # LARGE BODY PIPELINE (body_size == "L")
    # ============================================================
    else:  # body_size == "L"
        if diff == 0:
            # Large body + Large garment = Perfect fit
            return f"Perfect fit! Size {selected_size} matches your {body_size} body.", 1.0, "large_body"
        elif diff == -1:
            # Large body + Medium garment = Tight, doesn't fit well
            return f"Size {selected_size} will be tight on your {body_size} frame - may not fit comfortably.", 0.45, "large_body"
        elif diff == -2:
            # Large body + Small garment = WAY TOO SMALL - won't fit at all!
            return f"Size {selected_size} is WAY TOO SMALL for your {body_size} frame - extremely tight and short!", 0.25, "large_body"
        else:
            return f"Size {selected_size} on your {body_size} frame.", 1.0, "large_body"


def adjust_mask_for_size(mask, scale_factor, densepose_img=None, body_pipeline="small_body"):
    """
    Adjust the mask based on size selection with BODY-SIZE AWARE processing.

    SAFE mask adjustment with strict face protection.
    Different behaviors for different body types:

    SMALL BODY: Expand mask for loose/oversized look
    LARGE BODY: Aggressively erode + crop for "doesn't fit" look

    Args:
        mask: PIL Image of the mask
        scale_factor: How much to scale (>1 = larger/looser, <1 = smaller/tighter)
        densepose_img: Optional DensePose for guided dilation
        body_pipeline: "small_body", "medium_body", or "large_body"

    Returns:
        Adjusted mask as PIL Image
    """
    if abs(scale_factor - 1.0) < 0.01:
        print(f"  [Mask] No adjustment needed (scale={scale_factor})")
        return mask

    mask_np = np.array(mask)
    original_mask_coords = np.where(mask_np > 127)
    if len(original_mask_coords[0]) == 0:
        return mask

    # Store original boundaries - CRITICAL for face protection
    original_top = original_mask_coords[0].min()
    original_bottom = original_mask_coords[0].max()
    original_left = original_mask_coords[1].min()
    original_right = original_mask_coords[1].max()
    mask_height = original_bottom - original_top

    print(f"  [Mask] Original bounds: top={original_top}, bottom={original_bottom}, left={original_left}, right={original_right}")
    print(f"  [Mask] Body pipeline: {body_pipeline}")

    if scale_factor > 1.0:
        # ============================================================
        # LOOSE FIT: Expand mask ONLY horizontally and downward
        # Used for: Small body + Large garment
        # ============================================================

        h_kernel_size = int((scale_factor - 1.0) * 120) + 15
        h_kernel_size = max(15, min(h_kernel_size, 80))

        # Define COLLAR PROTECTION ZONE - top 15% of mask should NOT be dilated
        collar_zone = int(mask_height * 0.15)
        safe_top = original_top + collar_zone

        # Step 1: Create a copy for lower region dilation only
        lower_mask = mask_np.copy()
        lower_mask[:safe_top, :] = 0

        # Horizontal expansion on LOWER region only
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        dilated_lower = cv2.dilate(lower_mask, h_kernel, iterations=3)

        # Step 2: DOWNWARD-ONLY expansion
        v_expand = int((scale_factor - 1.0) * 60) + 15
        v_expand = max(15, min(v_expand, 80))

        expanded_down = np.zeros_like(dilated_lower)
        if v_expand < dilated_lower.shape[0]:
            expanded_down[v_expand:, :] = dilated_lower[:-v_expand, :]
        dilated_lower = cv2.bitwise_or(dilated_lower, expanded_down)

        # Step 3: Combine with ORIGINAL collar zone (undilated)
        adjusted = mask_np.copy()
        adjusted[safe_top:, :] = dilated_lower[safe_top:, :]

        # FINAL SAFETY: Zero out EVERYTHING above original_top
        adjusted[:original_top, :] = 0

        print(f"  [Mask] DILATED (safe): h={h_kernel_size}, v_down={v_expand}, collar_zone={collar_zone}px")

    else:
        # ============================================================
        # TIGHT FIT: Different strategies based on body pipeline
        # ============================================================

        if body_pipeline == "large_body":
            # LARGE BODY PIPELINE: NO MODIFICATIONS AT ALL
            # Return the original mask unchanged - let base IDM-VTON work
            print(f"  [Mask] LARGE BODY: Using ORIGINAL mask (NO modifications)")
            print(f"  [Mask] Scale factor {scale_factor} ignored for large body")
            adjusted = mask_np.copy()  # Keep original mask exactly as-is

        else:
            # SMALL/MEDIUM BODY PIPELINE: Standard erosion (less aggressive)
            kernel_size = int((1.0 - scale_factor) * 50) + 5
            kernel_size = max(5, min(kernel_size, 25))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            adjusted = cv2.erode(mask_np, kernel, iterations=1)

            print(f"  [Mask] ERODED for tight fit: kernel={kernel_size}")

    return Image.fromarray(adjusted)


def scale_garment_for_size(garment_img, scale_factor, body_pipeline="small_body"):
    """
    Scale the garment image to simulate different sizes with BODY-SIZE AWARE processing.

    This is the KEY technique that creates visible size differences:
    - For larger sizes (L): ZOOM INTO garment (crop center) - appears bigger on body
    - For smaller sizes (S): ZOOM OUT (add padding) - appears smaller on body

    BODY-SIZE AWARE:
    - Small body + Large garment: Gentle zoom (preserve design)
    - Large body + Small garment: AGGRESSIVE zoom out (garment too small!)

    Args:
        garment_img: PIL Image of the garment
        scale_factor: >1 for larger, <1 for smaller
        body_pipeline: "small_body", "medium_body", or "large_body"

    Returns:
        Scaled garment image (same dimensions as input)
    """
    if abs(scale_factor - 1.0) < 0.01:
        print(f"  [Garment Scale] No scaling needed (factor={scale_factor})")
        return garment_img

    print(f"  [Garment Scale] Applying scale factor: {scale_factor}, pipeline: {body_pipeline}")

    w, h = garment_img.size

    if scale_factor > 1.0:
        # ============================================================
        # LOOSE/OVERSIZED: Zoom in (crop center)
        # Used for: Small body + Large garment
        # ============================================================
        # GENTLE zoom to preserve garment design/print
        gentle_scale = 1.0 + (scale_factor - 1.0) * 0.3  # Only 30% of original scale
        crop_ratio = 1.0 / gentle_scale
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)

        # Crop from center (minimal crop to preserve design)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h

        cropped = garment_img.crop((left, top, right, bottom))
        scaled = cropped.resize((w, h), Image.LANCZOS)

        print(f"  [Garment Scale] Gentle zoom in: {gentle_scale:.2f} (preserving design)")

    else:
        # ============================================================
        # TIGHT FIT: Zoom out (add padding) - garment appears smaller
        # Different strategies based on body pipeline
        # ============================================================

        if body_pipeline == "large_body":
            # LARGE BODY PIPELINE: NO MODIFICATIONS AT ALL
            # Return the original garment unchanged - let base IDM-VTON work
            print(f"  [Garment Scale] LARGE BODY: Using ORIGINAL garment (NO modifications)")
            print(f"  [Garment Scale] Scale factor {scale_factor} ignored for large body")
            scaled = garment_img  # Keep original garment exactly as-is

        else:
            # SMALL/MEDIUM BODY PIPELINE: Standard shrinking
            amplified_scale = 1.0 - (1.0 - scale_factor) * 1.3  # 1.3x amplification
            new_w = int(w * amplified_scale)
            new_h = int(h * amplified_scale)

            # Resize garment smaller
            resized = garment_img.resize((new_w, new_h), Image.LANCZOS)

            # Create white background and paste garment in center
            scaled = Image.new('RGB', (w, h), (255, 255, 255))
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            scaled.paste(resized, (paste_x, paste_y))

            print(f"  [Garment Scale] Standard shrink: {amplified_scale:.2f}")

    return scaled


def get_size_prompt_modifier(body_size, selected_size):
    """
    Get prompt modification based on body vs selected size.

    BODY-SIZE AWARE prompts for stronger visual guidance.

    Small body + Large garment: Loose, oversized descriptions
    Large body + Small garment: Tight, stretched, doesn't fit descriptions
    """
    size_order = {"S": 0, "M": 1, "L": 2}
    diff = size_order[selected_size] - size_order[body_size]

    # ============================================================
    # SMALL BODY prompts (garment larger than body)
    # ============================================================
    if body_size == "S":
        if diff == 0:
            return "perfectly fitted tailored"
        elif diff == 1:
            return "loose relaxed comfortable with extra room"
        elif diff >= 2:
            return "very oversized baggy extremely loose draping billowing excess fabric"
        else:
            return "fitted"

    # ============================================================
    # MEDIUM BODY prompts
    # ============================================================
    elif body_size == "M":
        if diff == 0:
            return "perfectly fitted tailored"
        elif diff == 1:
            return "loose relaxed comfortable"
        elif diff == -1:
            return "fitted snug body-hugging"
        else:
            return "fitted"

    # ============================================================
    # LARGE BODY prompts - SIMPLIFIED to not confuse the model
    # ============================================================
    else:  # body_size == "L"
        # For large body, use simple prompts - let base model work naturally
        if diff == 0:
            return "fitted"
        elif diff == -1:
            return "fitted"  # Simple - don't confuse model
        elif diff == -2:
            return "fitted"  # Simple - don't confuse model
        else:
            return "fitted"

def get_negative_prompt_modifier(body_size, selected_size):
    """
    Get additional negative prompt terms for body-size aware generation.
    SIMPLIFIED for large body - no extra negative prompts.
    """
    # For large body, don't add any extra negative prompts
    # Let the base model work naturally
    if body_size == "L":
        return ""  # No modifications for large body

    # Only add modifiers for small/medium body
    size_order = {"S": 0, "M": 1, "L": 2}
    diff = size_order[selected_size] - size_order[body_size]

    if body_size == "S" and diff >= 2:
        return ", tight fitting, body hugging"
    else:
        return ""


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

print("=" * 60)
print("Loading IDM-VTON models...")
print("=" * 60)

print("\n[1/9] Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
print("✓ UNet loaded")

print("\n[2/9] Loading tokenizers...")
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
print("✓ Tokenizers loaded")

print("\n[3/9] Loading noise scheduler...")
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
print("✓ Noise scheduler loaded")

print("\n[4/9] Loading text encoders...")
text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
print("✓ Text encoders loaded")

print("\n[5/9] Loading image encoder...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
print("✓ Image encoder loaded")

print("\n[6/9] Loading VAE...")
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)
print("✓ VAE loaded")

print("\n[7/9] Loading UNet Encoder...")
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)
print("✓ UNet Encoder loaded")

print("\n[8/9] Loading parsing and openpose models...")
parsing_model = Parsing(0)
openpose_model = OpenPose(0)
print("✓ Parsing and OpenPose models loaded")

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

print("\n[9/9] Initializing pipeline...")
pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder
print("✓ Pipeline initialized")

print("\n" + "=" * 60)
print("All models loaded successfully!")
print("=" * 60)

def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed, selected_size):

    # ============================================================
    # INPUT VALIDATION
    # ============================================================
    if garm_img is None:
        raise ValueError("❌ Please upload a garment image before clicking Try-on!")

    if dict is None or dict.get("background") is None:
        raise ValueError("❌ Please upload a person image before clicking Try-on!")

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    # Get base directory (parent of gradio_demo)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
    model_path = os.path.join(base_dir, 'ckpt', 'densepose', 'model_final_162be9.pkl')

    args = apply_net.create_argument_parser().parse_args(('show', config_path, model_path, 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img_pil = Image.fromarray(pose_img).resize((768, 1024))

    # ============================================================
    # SIZE-AWARE PROCESSING (BODY-SIZE AWARE PIPELINES)
    # ============================================================
    # Calculate body size from DensePose
    body_size, torso_width, body_desc = calculate_body_size(pose_img)

    # Get size recommendation, scale factor, AND body pipeline
    size_message, scale_factor, body_pipeline = get_size_recommendation(body_size, selected_size)

    # Get prompt modifier for size (body-size aware)
    size_prompt = get_size_prompt_modifier(body_size, selected_size)

    # 1. SCALE GARMENT IMAGE (KEY for visible effect) - with body pipeline
    if scale_factor != 1.0:
        garm_img = scale_garment_for_size(garm_img, scale_factor, body_pipeline)

    # 2. ADJUST MASK (secondary effect) - with body pipeline
    if scale_factor != 1.0:
        mask = adjust_mask_for_size(mask, scale_factor, body_pipeline=body_pipeline)
        # Regenerate mask_gray with adjusted mask
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # Build size info display text with pipeline info
    if body_pipeline == "small_body":
        pipeline_info = """
SMALL BODY PIPELINE:
  S garment = Perfect Fit (1.0)
  M garment = Loose (1.30x dilation)
  L garment = Very Oversized (2.5x dilation)"""
    elif body_pipeline == "medium_body":
        pipeline_info = """
MEDIUM BODY PIPELINE:
  S garment = Tight (0.70x erosion)
  M garment = Perfect Fit (1.0)
  L garment = Loose (1.30x dilation)"""
    else:  # large_body
        pipeline_info = """
LARGE BODY PIPELINE:
  S garment = WAY TOO SMALL! (0.25x - extremely tight & short)
  M garment = Too Tight (0.45x - doesn't fit properly)
  L garment = Perfect Fit (1.0)"""

    size_info = f"""
╔══════════════════════════════════════════════════════╗
║  SIZE-AWARE VIRTUAL TRY-ON                           ║
╠══════════════════════════════════════════════════════╣
║  YOUR BODY SIZE: {body_size} (torso: {torso_width}px)
║  SELECTED GARMENT: {selected_size}
║  PIPELINE: {body_pipeline.upper().replace('_', ' ')}
╠══════════════════════════════════════════════════════╣
║  {size_message}
║
║  Scale Factor: {scale_factor:.2f}
║  Fit Style: "{size_prompt}"
╚══════════════════════════════════════════════════════╝
{pipeline_info}
"""

    # Print size info
    print(f"\n{'='*50}")
    print(f"SIZE-AWARE TRY-ON ({body_pipeline.upper()})")
    print(f"{'='*50}")
    print(f"Body Size Detected: {body_size} ({body_desc}, torso width: {torso_width}px)")
    print(f"Selected Size: {selected_size}")
    print(f"Body Pipeline: {body_pipeline}")
    print(f"Recommendation: {size_message}")
    print(f"Size Prompt: {size_prompt}")
    print(f"Scale Factor: {scale_factor}")
    print(f"{'='*50}\n")

    pose_img = pose_img_pil  # Use the PIL version
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 3. INCLUDE SIZE MODIFIER IN PROMPT
                prompt = f"model is wearing {size_prompt} " + garment_des

                # Get negative prompt modifier for body-size aware generation
                neg_modifier = get_negative_prompt_modifier(body_size, selected_size)
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality" + neg_modifier

                print(f"  [Prompt] Positive: {prompt}")
                print(f"  [Prompt] Negative modifier: {neg_modifier if neg_modifier else 'None'}")

                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality" + neg_modifier
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray, size_info
    else:
        return images[0], mask_gray, size_info

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON - Size-Aware Virtual Try-On 👕👔👚")
    gr.Markdown("Virtual Try-on with **SIZE SELECTION**! The system detects your body size and adjusts the garment fit accordingly.")

    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=False)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")

            # SIZE SELECTION DROPDOWN
            with gr.Row():
                selected_size = gr.Dropdown(
                    choices=["S", "M", "L"],
                    value="M",
                    label="Select Garment Size",
                    info="Choose the size you want to try on. System will detect your body size and adjust fit."
                )

            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)

        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
            # SIZE INFO DISPLAY
            size_info_display = gr.Textbox(
                label="Size Analysis",
                lines=8,
                interactive=False,
                show_copy_button=True
            )

        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)

    with gr.Column():
        try_button = gr.Button(value="Try-on", variant="primary")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

    # Updated click handler with size selection
    try_button.click(
        fn=start_tryon,
        inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed, selected_size],
        outputs=[image_out, masked_img, size_info_display],
        api_name='tryon'
    )

            


print("\n" + "=" * 60)
print("Starting Gradio server...")
print("=" * 60 + "\n")

# Get port from environment variable or use default
def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    return None

server_port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
available_port = find_available_port(server_port)

if available_port is None:
    print(f"Warning: Could not find available port starting from {server_port}")
    print("Trying to launch on any available port...")
    available_port = None  # Let Gradio find a port

if available_port and available_port != server_port:
    print(f"Port {server_port} is in use. Using port {available_port} instead.")

image_blocks.launch(share=False, show_api=False, server_name="0.0.0.0", server_port=available_port)

