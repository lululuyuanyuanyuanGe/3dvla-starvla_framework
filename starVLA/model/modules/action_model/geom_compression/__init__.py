"""Geometric feature compression modules for DiT action head injection.

Three approaches to compress [B, G, geom_dim] -> [B, K, hidden_dim]:
  - pool: Static prefix — shared linear + group pooling (Approach A)
  - per_layer: Per-layer projection — separate linear per DiT block (Approach B)
  - qformer: Q-Former — learned query cross-attention (Approach C)
"""

from .static_prefix import StaticPrefixCompressor
from .per_layer_projection import PerLayerProjectionCompressor
from .qformer_compression import QFormerCompressor


def build_geom_compressor(geom_dim: int, hidden_dim: int, config) -> "nn.Module":
    """Factory: build a geometric compressor from config.

    Args:
        geom_dim: MapAnything output feature dimension.
        hidden_dim: DiT cross-attention dimension (H_dit = vl_hidden_dim).
        config: Global config object. Reads from config.framework.action_model:
            - geom_compression_type: "pool" | "per_layer" | "qformer"
            - geom_dit_tokens: int (K, default 16)
            - For per_layer: uses num DiT layers from diffusion_model_cfg.num_layers

    Returns:
        nn.Module with forward(geom_feats) -> [B, K, hidden_dim]
        (PerLayerProjectionCompressor has forward(geom_feats, layer_idx) instead)
    """
    action_cfg = config.framework.action_model
    compression_type = getattr(action_cfg, "geom_compression_type", "pool")
    num_tokens_k = int(getattr(action_cfg, "geom_dit_tokens", 16))

    if compression_type == "pool":
        return StaticPrefixCompressor(
            geom_dim=geom_dim,
            hidden_dim=hidden_dim,
            num_tokens_k=num_tokens_k,
        )
    elif compression_type == "per_layer":
        diffusion_cfg = action_cfg.diffusion_model_cfg
        if isinstance(diffusion_cfg, dict):
            num_dit_layers = diffusion_cfg.get("num_layers", 16)
        else:
            num_dit_layers = getattr(diffusion_cfg, "num_layers", 16)
        return PerLayerProjectionCompressor(
            geom_dim=geom_dim,
            hidden_dim=hidden_dim,
            num_tokens_k=num_tokens_k,
            num_dit_layers=int(num_dit_layers),
        )
    elif compression_type == "qformer":
        num_heads = int(getattr(action_cfg, "geom_qformer_heads", 8))
        return QFormerCompressor(
            geom_dim=geom_dim,
            hidden_dim=hidden_dim,
            num_tokens_k=num_tokens_k,
            num_heads=num_heads,
        )
    else:
        raise ValueError(
            f"Unknown geom_compression_type={compression_type!r}. "
            f"Expected one of: pool, per_layer, qformer"
        )
