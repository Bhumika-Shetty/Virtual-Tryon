"""
Size Controller Module

CNN-based controller that generates spatial size guidance maps.
These maps provide per-pixel tight/loose guidance to the diffusion model.

Architecture:
- Input: Fused features from person + garment + size embedding
- Process: Multi-scale CNN with attention mechanisms
- Output: Spatial size map (H×W) with per-pixel guidance

The size map values indicate local fit tightness:
- 0.0 = very tight fit
- 0.33 = fitted
- 0.66 = loose
- 1.0 = very oversized/baggy

Author: Cinderella Team
Date: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SizeController(nn.Module):
    """
    CNN-based size controller for generating spatial size guidance maps.

    Takes fused person, garment, and size embedding features and produces
    a spatial map indicating where the garment should fit tightly vs loosely.
    """

    def __init__(
        self,
        in_channels: int = 512,  # Fused feature channels
        hidden_channels: int = 256,
        out_channels: int = 1,  # Single-channel size map
        num_layers: int = 4,
        use_attention: bool = True,
        output_size: Tuple[int, int] = (128, 96),  # Match latent size
    ):
        """
        Initialize the Size Controller.

        Args:
            in_channels: Input feature channels
            hidden_channels: Hidden layer channels
            out_channels: Output channels (1 for size map)
            num_layers: Number of convolutional layers
            use_attention: Whether to use self-attention
            output_size: Target output spatial size (H, W)
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.output_size = output_size

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(),
        )

        # Encoder: Downsample and extract features
        self.encoder_blocks = nn.ModuleList()
        current_channels = hidden_channels

        for i in range(num_layers // 2):
            self.encoder_blocks.append(
                ResidualBlock(
                    current_channels,
                    current_channels * 2 if i < num_layers // 2 - 1 else current_channels,
                    downsample=(i < num_layers // 2 - 1)
                )
            )
            if i < num_layers // 2 - 1:
                current_channels *= 2

        # Middle attention block (if enabled)
        if use_attention:
            self.attention = SpatialAttention(current_channels)
        else:
            self.attention = nn.Identity()

        # Decoder: Upsample and refine
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers // 2):
            out_ch = current_channels // 2 if i < num_layers // 2 - 1 else hidden_channels
            self.decoder_blocks.append(
                ResidualBlock(
                    current_channels,
                    out_ch,
                    upsample=(i < num_layers // 2 - 1)
                )
            )
            current_channels = out_ch

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        size_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the size controller.

        Args:
            fused_features: Tensor of shape (B, C, H, W) with fused person+garment features
            size_embedding: Optional size embedding (B, D) to modulate features

        Returns:
            size_map: Tensor of shape (B, 1, H, W) with spatial size guidance
        """
        x = fused_features

        # Project size embedding to spatial features (if provided)
        if size_embedding is not None:
            # Broadcast size embedding to spatial dimensions
            B, D = size_embedding.shape
            _, _, H, W = x.shape
            size_spatial = size_embedding.view(B, D, 1, 1).expand(B, D, H, W)

            # Concatenate with input features
            x = torch.cat([x, size_spatial], dim=1)

            # Project back to in_channels
            if not hasattr(self, 'size_fusion'):
                # Dynamically create fusion layer if needed
                self.size_fusion = nn.Conv2d(
                    self.in_channels + D, self.in_channels, kernel_size=1
                ).to(x.device)
            x = self.size_fusion(x)

        # Initial projection
        x = self.input_proj(x)

        # Encoder
        encoder_features = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_features.append(x)

        # Attention
        x = self.attention(x)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            # Add skip connection from encoder (reversed)
            if i < len(encoder_features) - 1:
                skip_idx = len(encoder_features) - 2 - i
                if x.shape[2:] == encoder_features[skip_idx].shape[2:]:
                    x = x + encoder_features[skip_idx]
            x = block(x)

        # Output projection
        size_map = self.output_proj(x)

        # Resize to target size if needed
        if size_map.shape[2:] != self.output_size:
            size_map = F.interpolate(
                size_map,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )

        return size_map


class ResidualBlock(nn.Module):
    """Residual block with optional up/downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        upsample: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.upsample = upsample

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()

        # Residual path
        if in_channels != out_channels or downsample or upsample:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        # Sampling layers
        if downsample:
            self.sample = nn.AvgPool2d(kernel_size=2, stride=2)
        elif upsample:
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.sample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Sample if needed
        out = self.sample(out)
        residual = self.sample(residual)

        # Residual connection
        residual = self.residual_proj(residual)
        out = out + residual
        out = self.activation(out)

        return out


class SpatialAttention(nn.Module):
    """Spatial self-attention module."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, H*W)
        out = out.reshape(B, C, H, W)

        # Project
        out = self.proj(out)

        # Residual
        return out + residual


class SimpleSizeController(nn.Module):
    """
    Simplified size controller for faster prototyping.

    Uses a lightweight CNN to generate size maps from size embeddings.
    """

    def __init__(
        self,
        size_embedding_dim: int = 768,
        output_size: Tuple[int, int] = (128, 96),
    ):
        super().__init__()

        self.size_embedding_dim = size_embedding_dim
        self.output_size = output_size

        # Simple MLP to expand size embedding
        self.mlp = nn.Sequential(
            nn.Linear(size_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size[0] * output_size[1]),
            nn.Sigmoid()
        )

    def forward(self, size_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            size_embedding: (B, D) size embedding

        Returns:
            size_map: (B, 1, H, W) spatial size map
        """
        B = size_embedding.shape[0]
        H, W = self.output_size

        # Generate flat size map
        size_map_flat = self.mlp(size_embedding)  # (B, H*W)

        # Reshape to spatial
        size_map = size_map_flat.view(B, 1, H, W)

        return size_map


def test_size_controller():
    """Test the size controller module."""
    print("Testing Size Controller Module")
    print("=" * 50)

    # Test full controller
    controller = SizeController(
        in_channels=512,
        hidden_channels=256,
        num_layers=4,
        use_attention=True,
        output_size=(128, 96)
    )

    print(f"Model parameters: {sum(p.numel() for p in controller.parameters()):,}")

    # Test forward pass
    batch_size = 2
    fused_features = torch.randn(batch_size, 512, 128, 96)
    size_embedding = torch.randn(batch_size, 768)

    size_map = controller(fused_features, size_embedding)

    print(f"\nInput features shape: {fused_features.shape}")
    print(f"Size embedding shape: {size_embedding.shape}")
    print(f"Output size map shape: {size_map.shape}")
    print(f"Size map range: [{size_map.min():.4f}, {size_map.max():.4f}]")
    print(f"Size map mean: {size_map.mean():.4f}")

    # Test simple controller
    print("\n" + "=" * 50)
    print("Testing Simple Size Controller")

    simple_controller = SimpleSizeController(
        size_embedding_dim=768,
        output_size=(128, 96)
    )

    print(f"Model parameters: {sum(p.numel() for p in simple_controller.parameters()):,}")

    simple_map = simple_controller(size_embedding)
    print(f"Simple size map shape: {simple_map.shape}")
    print(f"Simple size map range: [{simple_map.min():.4f}, {simple_map.max():.4f}]")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    test_size_controller()
