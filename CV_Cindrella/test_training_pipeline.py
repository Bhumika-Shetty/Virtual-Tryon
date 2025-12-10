"""
Test Training Pipeline

Verifies that the size-aware training code works correctly
without needing the full pretrained models.

Author: Cinderella Team
Date: 2025-12-08
"""

import os
import sys
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from src.size_embedder import SizeEmbedder, RelativeFitEmbedder, SinusoidalSizeEmbedder
        print("  ✓ size_embedder imports OK")
    except Exception as e:
        print(f"  ✗ size_embedder import failed: {e}")
        return False

    try:
        from src.size_aware_tryon_net import SizeAwareTryonNet
        print("  ✓ size_aware_tryon_net imports OK")
    except Exception as e:
        print(f"  ✗ size_aware_tryon_net import failed: {e}")
        return False

    return True


def test_size_embedder():
    """Test the SizeEmbedder module."""
    print("\nTesting SizeEmbedder...")

    from src.size_embedder import SizeEmbedder

    embedder = SizeEmbedder(
        num_body_sizes=3,
        num_cloth_sizes=4,
        embedding_dim=1280,
        use_relative_fit=True
    )

    # Test forward pass with all 9 combinations
    body_sizes = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])  # S, S, S, M, M, M, L, L, L
    cloth_sizes = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])  # S, M, L for each body

    output = embedder(body_sizes, cloth_sizes)

    print(f"  Input body sizes: {body_sizes.tolist()}")
    print(f"  Input cloth sizes: {cloth_sizes.tolist()}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: torch.Size([9, 1280])")

    assert output.shape == torch.Size([9, 1280]), "Shape mismatch!"
    print("  ✓ SizeEmbedder test passed")
    return True


def test_size_aware_unet():
    """Test the SizeAwareTryonNet wrapper with a mock UNet."""
    print("\nTesting SizeAwareTryonNet...")

    from src.size_embedder import SizeEmbedder
    from src.size_aware_tryon_net import SizeAwareTryonNet

    # Create mock UNet
    class MockUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {
                'time_embed_dim': 1280,
                'block_out_channels': [320, 640, 1280, 1280],
                'cross_attention_dim': 2048
            })()
            # Simple conv to make it a valid module
            self.conv = nn.Conv2d(13, 4, 1)

        def forward(self, sample, timestep, encoder_hidden_states,
                   added_cond_kwargs=None, garment_features=None, **kwargs):
            B = sample.shape[0]
            # Check that size_emb is in added_cond_kwargs
            if added_cond_kwargs and 'size_emb' in added_cond_kwargs:
                size_emb = added_cond_kwargs['size_emb']
                print(f"    Size embedding received: {size_emb.shape}")
            return type('Output', (), {'sample': torch.randn(B, 4, 128, 96)})()

    # Create size embedder
    size_embedder = SizeEmbedder(
        num_body_sizes=3,
        num_cloth_sizes=4,
        embedding_dim=1280
    )

    # Create size-aware wrapper
    mock_unet = MockUNet()
    size_aware_unet = SizeAwareTryonNet(
        base_unet=mock_unet,
        size_embedder=size_embedder,
        injection_method='added_cond'
    )

    # Test forward pass
    batch_size = 2
    sample = torch.randn(batch_size, 13, 128, 96)
    timestep = torch.tensor([100, 200])
    encoder_hidden_states = torch.randn(batch_size, 77, 2048)
    body_size = torch.tensor([0, 2])  # S, L
    cloth_size = torch.tensor([2, 0])  # L, S

    print(f"  Input sample shape: {sample.shape}")
    print(f"  Body sizes: {body_size.tolist()}")
    print(f"  Cloth sizes: {cloth_size.tolist()}")

    output = size_aware_unet(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        body_size=body_size,
        cloth_size=cloth_size,
    )

    print(f"  Output shape: {output.sample.shape}")
    print("  ✓ SizeAwareTryonNet test passed")
    return True


def test_dataset():
    """Test the dataset loader."""
    print("\nTesting Dataset...")

    # Import dataset class from training script
    try:
        # We'll create a minimal test without importing the full training script
        import json
        from PIL import Image

        data_dir = "/scratch/bds9746/CV_Vton_backup/CV_Cindrella/data_size_aware"

        # Check annotations file
        ann_path = os.path.join(data_dir, "size_annotations.json")
        with open(ann_path, 'r') as f:
            annotations = json.load(f)

        print(f"  Found {len(annotations['annotations'])} annotations")

        # Check train pairs
        pairs_path = os.path.join(data_dir, "train_pairs.txt")
        with open(pairs_path, 'r') as f:
            pairs = f.readlines()

        print(f"  Found {len(pairs)} training pairs")

        # Check an image file
        img_path = os.path.join(data_dir, "train/image/S_S.jpg")
        img = Image.open(img_path)
        print(f"  Sample image size: {img.size}")

        # Check cloth
        cloth_path = os.path.join(data_dir, "train/cloth/tshirt.jpg")
        cloth = Image.open(cloth_path)
        print(f"  Cloth image size: {cloth.size}")

        # Check densepose
        dp_path = os.path.join(data_dir, "train/image-densepose/S_S.jpg")
        dp = Image.open(dp_path)
        print(f"  DensePose image size: {dp.size}")

        # Check mask
        mask_path = os.path.join(data_dir, "train/agnostic-mask/S_S_mask.png")
        mask = Image.open(mask_path)
        print(f"  Mask image size: {mask.size}")

        print("  ✓ Dataset test passed")
        return True

    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False


def test_lora():
    """Test the LoRA implementation."""
    print("\nTesting LoRA...")

    try:
        # Define LoRA classes directly for isolated testing
        # (avoids importing from train_xl_size_aware which pulls in diffusers)

        class LoRALinear(nn.Module):
            """LoRA adapter for linear layers."""
            def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 1.0):
                super().__init__()
                self.original_layer = original_layer
                self.rank = rank
                self.alpha = alpha

                in_features = original_layer.in_features
                out_features = original_layer.out_features

                # LoRA matrices
                self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

                # Initialize A with random, B with zeros (so initial output is same as original)
                nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
                nn.init.zeros_(self.lora_B)

                self.scaling = alpha / rank

            def forward(self, x):
                result = self.original_layer(x)
                lora_result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
                return result + lora_result

        def add_lora_to_model(model, rank=16, alpha=1.0, target_modules=None):
            """Add LoRA adapters to specified modules."""
            lora_params = []

            if target_modules is None:
                target_modules = ['fc', 'linear', 'proj', 'query', 'key', 'value']

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    should_add = any(target in name.lower() for target in target_modules)
                    if should_add:
                        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]

                        if parent_name:
                            parent = dict(model.named_modules())[parent_name]
                        else:
                            parent = model

                        setattr(parent, child_name, lora_layer)
                        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

            return lora_params

        # Test LoRALinear directly
        original_linear = nn.Linear(512, 256)
        lora_layer = LoRALinear(original_linear, rank=16, alpha=16)

        test_input = torch.randn(2, 512)
        original_output = original_linear(test_input)
        lora_output = lora_layer(test_input)

        print(f"  Original output shape: {original_output.shape}")
        print(f"  LoRA output shape: {lora_output.shape}")

        # With zero-initialized B, outputs should be same initially
        diff = (original_output - lora_output).abs().max().item()
        print(f"  Initial output diff (should be ~0): {diff:.6f}")

        # Test add_lora_to_model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)

        model = SimpleModel()
        lora_params = add_lora_to_model(model, rank=8, alpha=8)
        print(f"  LoRA parameters added: {len(lora_params)}")
        print(f"  Total LoRA params: {sum(p.numel() for p in lora_params)}")

        # Verify model still works
        test_out = model.fc1(torch.randn(2, 128))
        print(f"  Model with LoRA output shape: {test_out.shape}")

        print("  ✓ LoRA test passed")
        return True

    except Exception as e:
        print(f"  ✗ LoRA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Size-Aware Training Pipeline Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("SizeEmbedder", test_size_embedder()))
    results.append(("SizeAwareTryonNet", test_size_aware_unet()))
    results.append(("Dataset", test_dataset()))
    results.append(("LoRA", test_lora()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 All tests passed! The training pipeline is ready.")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
