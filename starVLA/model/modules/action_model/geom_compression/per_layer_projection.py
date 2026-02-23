"""Approach B: Per-Layer Projection Compression.

Each DiT block gets geometry projected through a different learned linear layer,
allowing each layer to specialize in extracting different geometric aspects.
"""

import torch
import torch.nn as nn


class PerLayerProjectionCompressor(nn.Module):
    """Compress [B, G, geom_dim] -> per-layer [B, K, hidden_dim] via N separate projections.

    Args:
        geom_dim: Input geometric feature dimension (from MapAnything).
        hidden_dim: Target dimension matching DiT cross-attention (H_dit).
        num_tokens_k: Number of compressed output tokens (K).
        num_dit_layers: Number of DiT transformer blocks (N).
    """

    def __init__(self, geom_dim: int, hidden_dim: int, num_tokens_k: int = 16, num_dit_layers: int = 16):
        super().__init__()
        self.geom_dim = geom_dim
        self.hidden_dim = hidden_dim
        self.num_tokens_k = num_tokens_k
        self.num_dit_layers = num_dit_layers
        self.projectors = nn.ModuleList([
            nn.Linear(geom_dim, hidden_dim) for _ in range(num_dit_layers)
        ])

    def _group_pool(self, projected: torch.Tensor) -> torch.Tensor:
        """Pool G tokens down to K tokens via group averaging.

        Args:
            projected: [B, G, hidden_dim]

        Returns:
            [B, K, hidden_dim]
        """
        B, G, H = projected.shape
        K = self.num_tokens_k

        if G >= K:
            group_size = G // K
            remainder = G % K
            usable = group_size * K
            pooled = projected[:, :usable, :].reshape(B, K, group_size, H).mean(dim=2)
            if remainder > 0:
                leftover = projected[:, usable:, :].mean(dim=1, keepdim=True)
                pooled[:, -1:, :] = (pooled[:, -1:, :] * group_size + leftover * remainder) / (group_size + remainder)
        else:
            pooled = torch.zeros(B, K, H, device=projected.device, dtype=projected.dtype)
            pooled[:, :G, :] = projected

        return pooled

    def forward(self, geom_feats: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compress geometric features for a specific DiT layer.

        Args:
            geom_feats: [B, G, geom_dim] raw geometric features from MapAnything.
            layer_idx: Which DiT layer this is for (0..N-1).

        Returns:
            [B, K, hidden_dim] compressed geometric tokens for this specific layer.
        """
        # Step 1: Layer-specific projection
        projected = self.projectors[layer_idx](geom_feats)  # [B, G, hidden_dim]

        # Step 2: Group pool G -> K
        return self._group_pool(projected)  # [B, K, hidden_dim]
