"""Approach A: Static Prefix Compression.

Compresses G geometric tokens to K tokens via linear projection + group pooling.
The same K tokens are concatenated into the DiT cross-attention context at every layer.
"""

import torch
import torch.nn as nn


class StaticPrefixCompressor(nn.Module):
    """Compress [B, G, geom_dim] -> [B, K, hidden_dim] via projection + group pooling.

    Args:
        geom_dim: Input geometric feature dimension (from MapAnything).
        hidden_dim: Target dimension matching DiT cross-attention (H_dit).
        num_tokens_k: Number of compressed output tokens (K).
    """

    def __init__(self, geom_dim: int, hidden_dim: int, num_tokens_k: int = 16):
        super().__init__()
        self.geom_dim = geom_dim
        self.hidden_dim = hidden_dim
        self.num_tokens_k = num_tokens_k
        self.proj = nn.Linear(geom_dim, hidden_dim)

    def forward(self, geom_feats: torch.Tensor) -> torch.Tensor:
        """Compress geometric features to K tokens.

        Args:
            geom_feats: [B, G, geom_dim] raw geometric features from MapAnything.

        Returns:
            [B, K, hidden_dim] compressed geometric tokens.
        """
        B, G, _ = geom_feats.shape
        K = self.num_tokens_k

        # Step 1: Project geom_dim -> hidden_dim
        projected = self.proj(geom_feats)  # [B, G, hidden_dim]

        # Step 2: Group pool G -> K tokens
        if G >= K:
            # Split into K groups and mean-pool each group
            group_size = G // K
            remainder = G % K
            # Truncate to exact multiple for clean reshape
            usable = group_size * K
            pooled = projected[:, :usable, :].reshape(B, K, group_size, self.hidden_dim).mean(dim=2)
            # If there are leftover tokens, average them into the last group
            if remainder > 0:
                leftover = projected[:, usable:, :].mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
                pooled[:, -1:, :] = (pooled[:, -1:, :] * group_size + leftover * remainder) / (group_size + remainder)
        else:
            # G < K: pad with zeros (rare edge case)
            pooled = torch.zeros(B, K, self.hidden_dim, device=geom_feats.device, dtype=projected.dtype)
            pooled[:, :G, :] = projected

        return pooled  # [B, K, hidden_dim]
