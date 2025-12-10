#!/usr/bin/env python3
"""Download required checkpoints from HuggingFace for IDM-VTON"""

import os
from huggingface_hub import hf_hub_download
from pathlib import Path

# Repository and checkpoint paths
repo_id = "yisol/IDM-VTON"
base_dir = Path(__file__).parent / "ckpt"

# Checkpoints to download
checkpoints = {
    "densepose": [
        "densepose/model_final_162be9.pkl"
    ],
    "humanparsing": [
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx"
    ],
    "openpose": [
        "openpose/ckpts/body_pose_model.pth"
    ]
}

print("=" * 60)
print("Downloading IDM-VTON checkpoints from HuggingFace")
print("=" * 60)
print(f"Repository: {repo_id}")
print(f"Destination: {base_dir}")
print()

# Download each checkpoint
for category, files in checkpoints.items():
    print(f"\n[{category.upper()}]")
    for file_path in files:
        # Extract filename
        filename = os.path.basename(file_path)
        # Determine local directory
        if category == "densepose":
            local_dir = base_dir / "densepose"
        elif category == "humanparsing":
            local_dir = base_dir / "humanparsing"
        elif category == "openpose":
            local_dir = base_dir / "openpose" / "ckpts"
        
        local_path = local_dir / filename
        
        # Skip if already exists
        if local_path.exists():
            print(f"  ✓ {filename} already exists, skipping...")
            continue
        
        print(f"  Downloading {filename}...")
        try:
            # Download to a temp location first, then move to final location
            temp_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=str(base_dir / "temp"),
                local_dir_use_symlinks=False
            )
            # Move to final location
            import shutil
            local_dir.mkdir(parents=True, exist_ok=True)
            final_path = local_dir / filename
            shutil.move(temp_path, final_path)
            # Clean up temp directory if empty
            temp_dir = base_dir / "temp"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
            print(f"  ✓ Downloaded to {final_path}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")

print("\n" + "=" * 60)
print("Download complete!")
print("=" * 60)

