"""
Size-Aware TryonNet Wrapper

Wraps the base IDM-VTON UNet to inject size conditioning.
Supports multiple injection methods:
1. add_to_timestep - Add size embedding to timestep embedding (SDXL-style)
2. cross_attention - Concatenate size tokens to cross-attention sequence
3. added_cond - Add to SDXL's added_cond_kwargs

Author: Cinderella Team
Date: 2025-12-08
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union, List


class SizeAwareTryonNet(nn.Module):
    """
    Modified TryonNet with size conditioning injection.

    Wraps the base UNet and injects size embeddings during forward pass.
    """

    def __init__(
        self,
        base_unet: nn.Module,
        size_embedder: nn.Module,
        injection_method: str = 'added_cond',
        cross_attention_dim: int = 2048,  # SDXL cross-attention dim
    ):
        """
        Initialize Size-Aware TryonNet.

        Args:
            base_unet: The base IDM-VTON UNet model
            size_embedder: Size embedding module (SizeEmbedder, etc.)
            injection_method: How to inject size conditioning:
                - 'added_cond': Add to SDXL's added_cond_kwargs (recommended)
                - 'cross_attention': Concatenate to cross-attention sequence
                - 'timestep_add': Add to timestep embedding
            cross_attention_dim: Cross-attention dimension for token projection
        """
        super().__init__()

        self.unet = base_unet
        self.size_embedder = size_embedder
        self.injection_method = injection_method
        self.cross_attention_dim = cross_attention_dim

        # For cross-attention injection: project size embedding to sequence
        if injection_method == 'cross_attention':
            self.size_to_sequence = nn.Linear(
                size_embedder.embedding_dim,
                cross_attention_dim
            )
            nn.init.xavier_uniform_(self.size_to_sequence.weight)
            nn.init.zeros_(self.size_to_sequence.bias)

        # For timestep addition: project to timestep embedding dim
        if injection_method == 'timestep_add':
            # Get timestep embedding dimension from UNet config
            time_embed_dim = getattr(base_unet.config, 'time_embed_dim', 1280)
            self.size_to_timestep = nn.Linear(
                size_embedder.embedding_dim,
                time_embed_dim
            )
            nn.init.xavier_uniform_(self.size_to_timestep.weight)
            nn.init.zeros_(self.size_to_timestep.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        body_size: Optional[torch.Tensor] = None,
        cloth_size: Optional[torch.Tensor] = None,
        relative_fit: Optional[torch.Tensor] = None,
        size_embedding: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        garment_features: Optional[List[torch.Tensor]] = None,
        **kwargs
    ):
        """
        Forward pass with size conditioning.

        Args:
            sample: Noisy latent input (B, C, H, W)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings (B, seq_len, dim)
            body_size: (B,) body size indices (0=S, 1=M, 2=L)
            cloth_size: (B,) cloth size indices (0=S, 1=M, 2=L, 3=XL)
            relative_fit: (B,) optional relative fit values
            size_embedding: Pre-computed size embedding (overrides body/cloth_size)
            added_cond_kwargs: SDXL additional conditioning (text_embeds, time_ids)
            garment_features: Garment features from GarmentNet
            **kwargs: Additional arguments for base UNet

        Returns:
            UNet output with size conditioning applied
        """
        # Get size embedding
        if size_embedding is None and body_size is not None and cloth_size is not None:
            size_emb = self.size_embedder(body_size, cloth_size, relative_fit)
        elif size_embedding is not None:
            size_emb = size_embedding
        else:
            # No size conditioning - use null embedding
            batch_size = sample.shape[0]
            size_emb = self.size_embedder.get_null_embedding(batch_size, sample.device)

        # Initialize added_cond_kwargs if not provided
        if added_cond_kwargs is None:
            added_cond_kwargs = {}

        # Inject size conditioning based on method
        if self.injection_method == 'added_cond':
            # Method 1: Add to SDXL's added_cond_kwargs
            # This is similar to how SDXL handles size/crop conditioning
            added_cond_kwargs = dict(added_cond_kwargs)  # Don't modify original
            added_cond_kwargs['size_emb'] = size_emb

            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                garment_features=garment_features,
                **kwargs
            )

        elif self.injection_method == 'cross_attention':
            # Method 2: Concatenate size tokens to cross-attention sequence
            size_tokens = self.size_to_sequence(size_emb).unsqueeze(1)  # (B, 1, dim)
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, size_tokens],
                dim=1
            )

            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                garment_features=garment_features,
                **kwargs
            )

        elif self.injection_method == 'timestep_add':
            # Method 3: Add to timestep embedding
            # This requires modifying the UNet's time embedding flow
            # For now, we add it to added_cond_kwargs and expect UNet to handle it
            size_time = self.size_to_timestep(size_emb)
            added_cond_kwargs = dict(added_cond_kwargs)
            added_cond_kwargs['size_time_emb'] = size_time

            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                garment_features=garment_features,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown injection method: {self.injection_method}")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.unet.enable_gradient_checkpointing()

    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers attention for memory efficiency."""
        self.unet.enable_xformers_memory_efficient_attention()

    @property
    def config(self):
        """Return base UNet config."""
        return self.unet.config

    def parameters(self, recurse: bool = True):
        """Return all trainable parameters."""
        # Return both UNet and size embedder parameters
        yield from self.unet.parameters(recurse)
        yield from self.size_embedder.parameters(recurse)

        # Also yield injection layer parameters
        if hasattr(self, 'size_to_sequence'):
            yield from self.size_to_sequence.parameters(recurse)
        if hasattr(self, 'size_to_timestep'):
            yield from self.size_to_timestep.parameters(recurse)


class SizeConditionedUNet(nn.Module):
    """
    Alternative: Directly modify UNet to handle size conditioning in time embedding.

    This is a more integrated approach that modifies the time embedding block
    to incorporate size information.
    """

    def __init__(
        self,
        base_unet: nn.Module,
        size_embedding_dim: int = 1280,
    ):
        """
        Initialize Size-Conditioned UNet.

        Args:
            base_unet: Base UNet model
            size_embedding_dim: Dimension of size embeddings
        """
        super().__init__()

        self.unet = base_unet

        # Get time embedding dimension from UNet
        time_embed_dim = base_unet.config.block_out_channels[0] * 4

        # Size conditioning projection
        self.size_proj = nn.Sequential(
            nn.Linear(size_embedding_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Store original time embedding forward
        self._original_time_proj = base_unet.time_proj
        self._original_time_embedding = base_unet.time_embedding

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        size_embedding: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with size-conditioned time embedding."""
        # This is a simplified version - full implementation would need
        # to modify the UNet's internal forward pass
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            **kwargs
        )


def create_size_aware_unet(
    base_unet: nn.Module,
    size_embedder: nn.Module,
    injection_method: str = 'added_cond'
) -> SizeAwareTryonNet:
    """
    Factory function to create size-aware UNet.

    Args:
        base_unet: Base IDM-VTON UNet
        size_embedder: Size embedding module
        injection_method: Injection method ('added_cond', 'cross_attention', 'timestep_add')

    Returns:
        SizeAwareTryonNet wrapper
    """
    return SizeAwareTryonNet(
        base_unet=base_unet,
        size_embedder=size_embedder,
        injection_method=injection_method
    )


def test_size_aware_tryon_net():
    """Test the size-aware TryonNet wrapper."""
    print("Testing Size-Aware TryonNet")
    print("=" * 60)

    # Create mock UNet for testing
    class MockUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {
                'time_embed_dim': 1280,
                'block_out_channels': [320, 640, 1280, 1280]
            })()
            self.linear = nn.Linear(13 * 64 * 48, 4 * 64 * 48)

        def forward(self, sample, timestep, encoder_hidden_states,
                   added_cond_kwargs=None, garment_features=None, **kwargs):
            # Simple mock forward
            B = sample.shape[0]
            return type('Output', (), {'sample': torch.randn(B, 4, 64, 48)})()

    # Create mock size embedder
    from size_embedder import SizeEmbedder
    size_embedder = SizeEmbedder(embedding_dim=1280)

    # Create size-aware wrapper
    mock_unet = MockUNet()
    size_aware_net = SizeAwareTryonNet(
        base_unet=mock_unet,
        size_embedder=size_embedder,
        injection_method='added_cond'
    )

    print(f"Total parameters: {sum(p.numel() for p in size_aware_net.parameters()):,}")

    # Test forward pass
    batch_size = 2
    sample = torch.randn(batch_size, 13, 64, 48)
    timestep = torch.tensor([100, 200])
    encoder_hidden_states = torch.randn(batch_size, 77, 2048)
    body_size = torch.tensor([0, 2])  # S, L
    cloth_size = torch.tensor([2, 0])  # L, S

    output = size_aware_net(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        body_size=body_size,
        cloth_size=cloth_size,
    )

    print(f"\nInput sample shape: {sample.shape}")
    print(f"Output sample shape: {output.sample.shape}")
    print(f"Body sizes: {body_size.tolist()}")
    print(f"Cloth sizes: {cloth_size.tolist()}")

    print("\n" + "=" * 60)
    print("Test passed!")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    test_size_aware_tryon_net()
