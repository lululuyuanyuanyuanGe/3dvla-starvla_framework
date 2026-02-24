"""Standalone MapAnything geometric encoder wrapper.

Wraps the MapAnything model (DINOv2 encoder + multi-view cross-attention transformer)
as an nn.Module that takes images and returns per-location 3D geometric features.

Input:  pixel_values [B, V, 3, H, W] (float, [0,1] range)
        intrinsics   [B, V, 3, 3] or None
Output: .last_hidden_state [B, G, geom_dim]
        where G = spatial sequence length, geom_dim ~ 1024

Used by QwenMapAnythingPI to extract geometry for direct injection into DiT action head.
"""

import os
import sys
import torch
import torch.nn as nn

# MapAnything library lives under mapanything_llava3d/model/map-anything/
_LOCAL_MAPANYTHING_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",  # up to starVLA/
    "mapanything_llava3d", "model", "map-anything",
)
_LOCAL_MAPANYTHING_ROOT = os.path.normpath(_LOCAL_MAPANYTHING_ROOT)
if _LOCAL_MAPANYTHING_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_MAPANYTHING_ROOT)

# If mapanything was already imported from elsewhere, force reload from local path.
if "mapanything" in sys.modules:
    mod = sys.modules["mapanything"]
    mod_file = getattr(mod, "__file__", "") or ""
    if not mod_file.startswith(_LOCAL_MAPANYTHING_ROOT):
        del sys.modules["mapanything"]

from mapanything.models.mapanything.model import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput


class _GeomOutput:
    """Simple output container matching HuggingFace convention."""
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MapAnythingWrapper(nn.Module):
    """Standalone wrapper around MapAnything for 3D geometric feature extraction.

    Pipeline:
        1. Split multi-view images into per-view dicts
        2. Encode each view with DINOv2 (_encode_n_views)
        3. Fuse views with multi-view cross-attention transformer (info_sharing)
        4. Reshape [B, C, H, W] feature maps to sequence [B, G, geom_dim]

    Args:
        config: object with `mapanything_model_name_or_path` attribute
    """

    def __init__(self, config):
        super().__init__()
        self.map_anything_model = MapAnything.from_pretrained(
            config.mapanything_model_name_or_path
        )
        enc_dim = getattr(self.map_anything_model.encoder, "enc_embed_dim", None)

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.hidden_size = int(enc_dim) if enc_dim is not None else 1024

    @staticmethod
    def _unwrap_feature(feature):
        """Handle different MapAnything output formats (tuple, dict, tensor)."""
        if isinstance(feature, tuple):
            if len(feature) == 0:
                return feature
            return feature[0]
        if isinstance(feature, dict):
            for k in ("features", "feature", "x"):
                if k in feature:
                    return feature[k]
        return feature

    @staticmethod
    def _ensure_4d_feature(feature, view_idx: int | None = None):
        """Ensure feature is [B, C, H, W] format."""
        if not isinstance(feature, torch.Tensor):
            raise TypeError(
                f"map-anything view[{view_idx}] is not a Tensor: {type(feature)}"
            )
        if feature.dim() == 4:
            return feature
        if feature.dim() == 3:
            b, c, l = feature.shape
            if l == 1:
                return feature.view(b, c, 1, 1)
            side = int(l**0.5)
            if side * side == l:
                return feature.view(b, c, side, side)
        raise ValueError(
            f"map-anything view[{view_idx}] has unsupported shape {tuple(feature.shape)}; "
            "expected 4D (N,C,H,W) or 3D with L=1 or perfect square."
        )

    def forward(self, pixel_values, intrinsics):
        """Extract 3D geometric features from multi-view images.

        Args:
            pixel_values: [B, V, 3, H, W] or [B, 3, H, W] float tensor in [0,1]
            intrinsics: [B, V, 3, 3] camera intrinsics or None

        Returns:
            _GeomOutput with .last_hidden_state: [B, G, geom_dim]
        """
        views = []
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            b, v = pixel_values.shape[:2]
            for view_idx in range(v):
                view = {
                    "img": pixel_values[:, view_idx].float().contiguous(),
                    "data_norm_type": ["dinov2"],
                }
                if intrinsics is not None:
                    if isinstance(intrinsics, torch.Tensor) and intrinsics.dim() == 4:
                        view_intrinsics = intrinsics[:, view_idx]
                    else:
                        view_intrinsics = intrinsics
                    view["intrinsics"] = view_intrinsics.float().contiguous()
                views.append(view)
        else:
            view = {
                "img": pixel_values.float().contiguous(),
                "data_norm_type": ["dinov2"],
            }
            if intrinsics is not None:
                view["intrinsics"] = intrinsics.float().contiguous()
            views = [view]

        # Step 1: Per-view DINOv2 encoding
        all_encoder_features = self.map_anything_model._encode_n_views(views)
        if isinstance(all_encoder_features, tuple) and len(all_encoder_features) == 2:
            all_encoder_features, _registers = all_encoder_features

        if isinstance(all_encoder_features, (list, tuple)):
            converted = []
            for i, f in enumerate(all_encoder_features):
                f = self._unwrap_feature(f)
                f = self._ensure_4d_feature(f, view_idx=i)
                converted.append(f)
            all_encoder_features = converted

        # Step 2: Multi-view cross-attention fusion
        info_sharing_input = MultiViewTransformerInput(features=all_encoder_features)
        final_features, _ = self.map_anything_model.info_sharing(info_sharing_input)

        # Step 3: Reshape to sequence format [B, G, geom_dim]
        if len(final_features.features) == 1:
            geometric_features = final_features.features[0]
            if geometric_features.dim() == 4:
                b, c, h, w = geometric_features.shape
                geometric_features = geometric_features.permute(0, 2, 3, 1).reshape(
                    b, h * w, c
                )
        else:
            seq_features = []
            for f in final_features.features:
                if f.dim() == 4:
                    b, c, h, w = f.shape
                    f = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
                elif f.dim() == 3:
                    pass
                else:
                    raise ValueError(
                        f"Unsupported feature shape: {tuple(f.shape)}"
                    )
                seq_features.append(f)
            geometric_features = torch.cat(seq_features, dim=1)

        return _GeomOutput(geometric_features)
