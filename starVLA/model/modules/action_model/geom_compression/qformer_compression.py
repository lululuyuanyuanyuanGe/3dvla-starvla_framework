"""Approach C: Q-Former Compression.

K learnable query tokens cross-attend over all G geometric tokens to produce
K adaptive compressed tokens. The attention weights change per scene, allowing
each query to specialize in extracting different geometric aspects.
"""

import torch
import torch.nn as nn


class QFormerCompressor(nn.Module):
    """Compress [B, G, geom_dim] -> [B, K, hidden_dim] via learned query cross-attention.

    Args:
        geom_dim: Input geometric feature dimension (from MapAnything).
        hidden_dim: Target dimension matching DiT cross-attention (H_dit).
        num_tokens_k: Number of learnable query tokens / compressed output tokens (K).
        num_heads: Number of attention heads in the cross-attention module.
    """

    def __init__(self, geom_dim: int, hidden_dim: int, num_tokens_k: int = 16, num_heads: int = 8):
        super().__init__()
        self.geom_dim = geom_dim
        self.hidden_dim = hidden_dim
        self.num_tokens_k = num_tokens_k
        self.num_heads = num_heads

        # Project geometric features to hidden_dim
        self.geom_proj = nn.Linear(geom_dim, hidden_dim)

        # K learnable query tokens
        self.geom_queries = nn.Parameter(torch.randn(1, num_tokens_k, hidden_dim) * 0.02)

        # Cross-attention: queries attend over projected geometric keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm for stability
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

    def forward(self, geom_feats: torch.Tensor) -> torch.Tensor:
        """Compress geometric features via learned query cross-attention.

        Args:
            geom_feats: [B, G, geom_dim] raw geometric features from MapAnything.

        Returns:
            [B, K, hidden_dim] compressed geometric tokens.
        """
        B = geom_feats.shape[0]

        # Step 1: Project geom_dim -> hidden_dim
        geom_projected = self.geom_proj(geom_feats)  # [B, G, hidden_dim]
        geom_projected = self.kv_norm(geom_projected)

        # Step 2: Expand learnable queries to batch size
        queries = self.geom_queries.expand(B, -1, -1)  # [B, K, hidden_dim]
        queries = self.query_norm(queries)

        # Step 3: Cross-attention — queries attend over all G geometric tokens
        compressed, _ = self.cross_attn(
            query=queries,           # [B, K, hidden_dim]
            key=geom_projected,      # [B, G, hidden_dim]
            value=geom_projected,    # [B, G, hidden_dim]
        )  # -> [B, K, hidden_dim]

        return compressed  # [B, K, hidden_dim]
