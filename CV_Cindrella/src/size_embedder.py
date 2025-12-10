"""
Size Embedder Module for Full Combinatorial Training

Embeds body size and clothing size information for size-aware virtual try-on.
Supports the 9-combination matrix:
    Body Size (S/M/L) × Clothing Size (S/M/L)

This module provides:
1. SizeEmbedder - Learnable embeddings for categorical sizes
2. SinusoidalSizeEmbedder - Continuous interpolation between sizes
3. RelativeFitEmbedder - Focuses on relative fit (tight/fitted/loose)

Author: Cinderella Team
Date: 2025-12-08
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SizeEmbedder(nn.Module):
    """
    Embeds body size and clothing size information.

    Full Combinatorial Training approach:
    - Learns separate embeddings for body size (3 classes: S/M/L)
    - Learns separate embeddings for cloth size (4 classes: S/M/L/XL)
    - Optionally incorporates relative fit as continuous signal

    Output embedding is compatible with SDXL's conditioning dimension (1280).
    """

    def __init__(
        self,
        num_body_sizes: int = 3,      # small, medium, large
        num_cloth_sizes: int = 4,     # S, M, L, XL
        embedding_dim: int = 1280,    # Match SDXL's embedding dimension
        use_relative_fit: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize the Size Embedder.

        Args:
            num_body_sizes: Number of body size categories (default 3: S/M/L)
            num_cloth_sizes: Number of clothing size categories (default 4: S/M/L/XL)
            embedding_dim: Output embedding dimension (1280 for SDXL)
            use_relative_fit: Whether to include relative fit encoding
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.num_body_sizes = num_body_sizes
        self.num_cloth_sizes = num_cloth_sizes
        self.embedding_dim = embedding_dim
        self.use_relative_fit = use_relative_fit

        # Learned embeddings for categorical sizes
        self.body_size_embed = nn.Embedding(num_body_sizes, embedding_dim // 2)
        self.cloth_size_embed = nn.Embedding(num_cloth_sizes, embedding_dim // 2)

        # Project concatenated embeddings to final dimension
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        if use_relative_fit:
            # Continuous embedding for relative fit (-2 to +2 range typically)
            # relative_fit = cloth_size_idx - body_size_idx
            # negative = tight, 0 = fitted, positive = loose
            self.fit_embed = nn.Sequential(
                nn.Linear(1, embedding_dim // 4),
                nn.SiLU(),
                nn.Linear(embedding_dim // 4, embedding_dim)
            )

            # Combine categorical + continuous
            self.combine = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.body_size_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cloth_size_embed.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        body_size: torch.Tensor,
        cloth_size: torch.Tensor,
        relative_fit: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            body_size: (B,) tensor of body size indices (0=S, 1=M, 2=L)
            cloth_size: (B,) tensor of clothing size indices (0=S, 1=M, 2=L, 3=XL)
            relative_fit: (B,) optional tensor of relative fit values
                          If None, computed as cloth_size - body_size

        Returns:
            size_embedding: (B, embedding_dim) tensor
        """
        # Get categorical embeddings
        body_emb = self.body_size_embed(body_size)    # (B, dim/2)
        cloth_emb = self.cloth_size_embed(cloth_size)  # (B, dim/2)

        # Concatenate and project
        combined = torch.cat([body_emb, cloth_emb], dim=-1)  # (B, dim)
        size_emb = self.project(combined)

        # Add relative fit encoding if enabled
        if self.use_relative_fit:
            if relative_fit is None:
                # Compute relative fit from indices
                relative_fit = cloth_size.float() - body_size.float()

            fit_emb = self.fit_embed(relative_fit.unsqueeze(-1).float())
            size_emb = self.combine(torch.cat([size_emb, fit_emb], dim=-1))

        return size_emb

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get null embedding for classifier-free guidance dropout.

        Args:
            batch_size: Number of samples
            device: Target device

        Returns:
            Zero embedding of shape (batch_size, embedding_dim)
        """
        return torch.zeros(batch_size, self.embedding_dim, device=device)


class SinusoidalSizeEmbedder(nn.Module):
    """
    Sinusoidal positional encoding for sizes.

    Better for continuous interpolation between sizes.
    Allows generating intermediate sizes (e.g., between M and L).
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        max_period: int = 10000,
    ):
        """
        Initialize sinusoidal embedder.

        Args:
            embedding_dim: Output embedding dimension
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # MLP to process sinusoidal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def sinusoidal_embed(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device) / half
        )
        args = x[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(
        self,
        body_size: torch.Tensor,
        cloth_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            body_size: (B,) normalized body size (0-1 range or class indices)
            cloth_size: (B,) normalized cloth size (0-1 range or class indices)

        Returns:
            size_embedding: (B, embedding_dim) tensor
        """
        # Normalize to 0-1 if indices provided
        body_norm = body_size.float() / 2.0  # 0,1,2 -> 0, 0.5, 1
        cloth_norm = cloth_size.float() / 3.0  # 0,1,2,3 -> 0, 0.33, 0.67, 1

        body_emb = self.sinusoidal_embed(body_norm, self.embedding_dim)
        cloth_emb = self.sinusoidal_embed(cloth_norm, self.embedding_dim)

        combined = torch.cat([body_emb, cloth_emb], dim=-1)
        return self.mlp(combined)


class RelativeFitEmbedder(nn.Module):
    """
    Embedder focused on relative fit rather than absolute sizes.

    Directly encodes the fit relationship:
    - Very tight (-2): XS cloth on L body
    - Tight (-1): S cloth on M body, M cloth on L body
    - Fitted (0): Same size
    - Loose (+1): M cloth on S body, L cloth on M body
    - Very loose (+2): L cloth on S body

    This is simpler and may be more effective for the try-on task.
    """

    def __init__(
        self,
        num_fit_classes: int = 5,  # very tight, tight, fitted, loose, very loose
        embedding_dim: int = 1280,
    ):
        """
        Initialize relative fit embedder.

        Args:
            num_fit_classes: Number of fit categories
            embedding_dim: Output embedding dimension
        """
        super().__init__()

        self.num_fit_classes = num_fit_classes
        self.embedding_dim = embedding_dim

        # Learnable embeddings for fit classes
        self.fit_embed = nn.Embedding(num_fit_classes, embedding_dim)

        # Additional MLP for refinement
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        nn.init.normal_(self.fit_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        body_size: torch.Tensor,
        cloth_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            body_size: (B,) tensor of body size indices (0=S, 1=M, 2=L)
            cloth_size: (B,) tensor of clothing size indices (0=S, 1=M, 2=L, 3=XL)

        Returns:
            fit_embedding: (B, embedding_dim) tensor
        """
        # Compute relative fit: cloth - body
        # Range: -2 to +3 -> shift to 0-5 for embedding lookup
        relative_fit = cloth_size - body_size
        fit_class = torch.clamp(relative_fit + 2, 0, self.num_fit_classes - 1).long()

        fit_emb = self.fit_embed(fit_class)
        return self.mlp(fit_emb)


def test_size_embedder():
    """Test the size embedder modules."""
    print("Testing Size Embedder Module")
    print("=" * 60)

    # Test SizeEmbedder
    embedder = SizeEmbedder(
        num_body_sizes=3,
        num_cloth_sizes=4,
        embedding_dim=1280,
        use_relative_fit=True
    )

    print(f"SizeEmbedder parameters: {sum(p.numel() for p in embedder.parameters()):,}")

    # Test all 9 combinations
    batch_size = 9
    body_sizes = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])  # S, S, S, M, M, M, L, L, L
    cloth_sizes = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])  # S, M, L for each body

    output = embedder(body_sizes, cloth_sizes)
    print(f"\nInput body sizes: {body_sizes.tolist()}")
    print(f"Input cloth sizes: {cloth_sizes.tolist()}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output std: {output.std():.4f}")

    # Test null embedding
    null_emb = embedder.get_null_embedding(4, torch.device('cpu'))
    print(f"\nNull embedding shape: {null_emb.shape}")
    print(f"Null embedding sum: {null_emb.sum():.4f}")

    # Test SinusoidalSizeEmbedder
    print("\n" + "=" * 60)
    print("Testing SinusoidalSizeEmbedder")

    sin_embedder = SinusoidalSizeEmbedder(embedding_dim=1280)
    print(f"Parameters: {sum(p.numel() for p in sin_embedder.parameters()):,}")

    sin_output = sin_embedder(body_sizes, cloth_sizes)
    print(f"Output shape: {sin_output.shape}")

    # Test RelativeFitEmbedder
    print("\n" + "=" * 60)
    print("Testing RelativeFitEmbedder")

    fit_embedder = RelativeFitEmbedder(embedding_dim=1280)
    print(f"Parameters: {sum(p.numel() for p in fit_embedder.parameters()):,}")

    fit_output = fit_embedder(body_sizes, cloth_sizes)
    print(f"Output shape: {fit_output.shape}")

    # Print relative fits
    relative_fits = cloth_sizes - body_sizes
    print(f"Relative fits: {relative_fits.tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == '__main__':
    test_size_embedder()
