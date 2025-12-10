"""
Size Encoder Module

MLP-based encoder that maps size ratios to high-dimensional embeddings.
These embeddings are injected into the UNet's cross-attention layers.

Architecture:
- Input: [width_ratio, length_ratio, shoulder_ratio] (3-dim)
- Hidden: 2-3 layer MLP with ReLU/GELU activation
- Output: 768-dim embedding (compatible with CLIP embedding dimension)

Author: Cinderella Team
Date: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SizeEncoder(nn.Module):
    """
    MLP encoder for size conditioning.

    Maps 3D size ratios to 768-dim embeddings that can be injected
    into the diffusion model's cross-attention layers.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 3,
        activation: str = 'gelu',
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Initialize the Size Encoder.

        Args:
            input_dim: Input dimension (default 3 for width/length/shoulder ratios)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (768 for CLIP compatibility)
            num_layers: Number of MLP layers (2 or 3 recommended)
            activation: Activation function ('relu', 'gelu', 'silu')
            dropout: Dropout probability for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP layers
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        size_ratios: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the size encoder.

        Args:
            size_ratios: Tensor of shape (batch_size, 3) containing
                        [width_ratio, length_ratio, shoulder_ratio]
            return_intermediate: If True, return intermediate features

        Returns:
            size_embedding: Tensor of shape (batch_size, output_dim)
        """
        # Ensure input is 2D
        if size_ratios.dim() == 1:
            size_ratios = size_ratios.unsqueeze(0)

        assert size_ratios.shape[-1] == self.input_dim, \
            f"Expected input dim {self.input_dim}, got {size_ratios.shape[-1]}"

        # Normalize ratios to a reasonable range (0.5 to 2.0 -> -1 to 1)
        # This helps with training stability
        normalized_ratios = (size_ratios - 1.0) / 0.5
        normalized_ratios = torch.clamp(normalized_ratios, -2.0, 2.0)

        # Forward through MLP
        size_embedding = self.mlp(normalized_ratios)

        return size_embedding

    def encode_batch(
        self,
        width_ratios: torch.Tensor,
        length_ratios: torch.Tensor,
        shoulder_ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method to encode separate ratio tensors.

        Args:
            width_ratios: (batch_size,) tensor
            length_ratios: (batch_size,) tensor
            shoulder_ratios: (batch_size,) tensor

        Returns:
            size_embedding: (batch_size, output_dim) tensor
        """
        size_ratios = torch.stack([width_ratios, length_ratios, shoulder_ratios], dim=1)
        return self.forward(size_ratios)


class SizeEmbeddingLayer(nn.Module):
    """
    Alternative: Learnable embedding layer for discrete size labels.

    This can be used instead of or in addition to the MLP encoder
    for experiments with discrete size categories.
    """

    def __init__(
        self,
        num_size_classes: int = 4,  # tight, fitted, loose, oversized
        embedding_dim: int = 768,
    ):
        """
        Initialize the size embedding layer.

        Args:
            num_size_classes: Number of discrete size categories
            embedding_dim: Embedding dimension
        """
        super().__init__()

        self.num_size_classes = num_size_classes
        self.embedding_dim = embedding_dim

        # Learnable embeddings for each size class
        self.size_embeddings = nn.Embedding(num_size_classes, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.size_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, size_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            size_labels: (batch_size,) tensor of size class IDs (0-3)

        Returns:
            size_embedding: (batch_size, embedding_dim) tensor
        """
        return self.size_embeddings(size_labels)


class HybridSizeEncoder(nn.Module):
    """
    Hybrid encoder combining continuous ratios and discrete labels.

    Uses both the MLP encoder for continuous ratios and embedding layer
    for discrete labels, then fuses them.
    """

    def __init__(
        self,
        ratio_dim: int = 3,
        num_size_classes: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 768,
        fusion_method: str = 'add'  # 'add', 'concat', 'gated'
    ):
        """
        Initialize hybrid encoder.

        Args:
            ratio_dim: Dimension of continuous ratios
            num_size_classes: Number of discrete size classes
            hidden_dim: Hidden dimension for MLP
            output_dim: Output embedding dimension
            fusion_method: How to fuse continuous and discrete features
        """
        super().__init__()

        self.fusion_method = fusion_method

        # Continuous ratio encoder
        self.ratio_encoder = SizeEncoder(
            input_dim=ratio_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

        # Discrete label encoder
        self.label_encoder = SizeEmbeddingLayer(
            num_size_classes=num_size_classes,
            embedding_dim=output_dim
        )

        # Fusion layer (if needed)
        if fusion_method == 'concat':
            self.fusion_proj = nn.Linear(output_dim * 2, output_dim)
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )

    def forward(
        self,
        size_ratios: torch.Tensor,
        size_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with both continuous and discrete inputs.

        Args:
            size_ratios: (batch_size, 3) continuous ratios
            size_labels: (batch_size,) discrete labels

        Returns:
            size_embedding: (batch_size, output_dim) fused embedding
        """
        # Encode both modalities
        ratio_emb = self.ratio_encoder(size_ratios)
        label_emb = self.label_encoder(size_labels)

        # Fuse embeddings
        if self.fusion_method == 'add':
            output = ratio_emb + label_emb
        elif self.fusion_method == 'concat':
            output = torch.cat([ratio_emb, label_emb], dim=1)
            output = self.fusion_proj(output)
        elif self.fusion_method == 'gated':
            gate_input = torch.cat([ratio_emb, label_emb], dim=1)
            gate = self.gate(gate_input)
            output = gate * ratio_emb + (1 - gate) * label_emb
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return output


def test_size_encoder():
    """Test the size encoder module."""
    print("Testing Size Encoder Module")
    print("=" * 50)

    # Create encoder
    encoder = SizeEncoder(
        input_dim=3,
        hidden_dim=256,
        output_dim=768,
        num_layers=3
    )

    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    batch_size = 4
    size_ratios = torch.tensor([
        [0.85, 0.95, 0.90],  # tight
        [1.0, 1.05, 1.0],     # fitted
        [1.15, 1.2, 1.18],    # loose
        [1.4, 1.35, 1.42],    # oversized
    ])

    output = encoder(size_ratios)
    print(f"\nInput shape: {size_ratios.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output std: {output.std():.4f}")

    # Test discrete embedding layer
    print("\n" + "=" * 50)
    print("Testing Discrete Size Embedding Layer")

    label_encoder = SizeEmbeddingLayer(num_size_classes=4, embedding_dim=768)
    size_labels = torch.tensor([0, 1, 2, 3])  # tight, fitted, loose, oversized

    label_output = label_encoder(size_labels)
    print(f"Label input shape: {size_labels.shape}")
    print(f"Label output shape: {label_output.shape}")

    # Test hybrid encoder
    print("\n" + "=" * 50)
    print("Testing Hybrid Size Encoder")

    hybrid_encoder = HybridSizeEncoder(fusion_method='gated')
    hybrid_output = hybrid_encoder(size_ratios, size_labels)
    print(f"Hybrid output shape: {hybrid_output.shape}")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    test_size_encoder()
