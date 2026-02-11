import torch
import torch.nn as nn
from typing import Tuple

class GeometryFusion(nn.Module):
    """
    Fuses 2D Semantic Features (SigLIP) with 3D Geometric Features (MapAnything).
    
    Strategy: Geometric Pooling
    1. Resize/Pool 3D Points [B, 518, 518, 3] -> [B, 27, 27, 3] (matching SigLIP grid)
    2. Flatten to [B, 729, 3]
    3. Project Semantics & Geometry to same dim
    4. Add and Project to LLM dim
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if isinstance(config, dict):
            proj_config = config.get("framework", {}).get("projector", {})
        else:
            proj_config = config.framework.projector

        self.semantic_dim = getattr(proj_config, "semantic_dim", 1152) # SigLIP Large
        self.geometry_dim = getattr(proj_config, "geometry_dim", 3)    # x,y,z
        self.llm_dim = getattr(proj_config, "llm_dim", 4096)           # LLaVA-7B
        self.fusion_dim = getattr(proj_config, "fusion_dim", 1152)     # Internal fusion dim
        
        # Grid settings
        self.target_grid_size = 27 # 384 / 14 = 27.4 -> 27 patches
        
        # Pooling for Geometry
        self.pool = nn.AdaptiveAvgPool2d((self.target_grid_size, self.target_grid_size))
        
        # Projections
        self.semantic_proj = nn.Linear(self.semantic_dim, self.fusion_dim)
        
        # Geometry MLP (Positional Encoding-like)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.geometry_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
        
        # Final Projection to LLM
        self.output_proj = nn.Linear(self.fusion_dim, self.llm_dim)
        
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

    def forward(self, semantic_features: torch.Tensor, geometric_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_features: [B, 729, 1152] - Flattened patch features from SigLIP
            geometric_map: [B, 518, 518, 3] - Dense point map from MapAnything
            
        Returns:
            fused_features: [B, 729, 4096] - Embeddings ready for LLaVA
        """
        B = semantic_features.shape[0]
        
        # 1. Process Geometry
        # Permute to [B, 3, H, W] for pooling
        geo = geometric_map.permute(0, 3, 1, 2) 
        # Pool to [B, 3, 27, 27]
        geo_pooled = self.pool(geo)
        # Permute back and flatten: [B, 729, 3]
        geo_flat = geo_pooled.permute(0, 2, 3, 1).reshape(B, -1, self.geometry_dim)
        
        # 2. Embeddings
        sem_emb = self.semantic_proj(semantic_features) # [B, 729, F]
        geo_emb = self.geometry_mlp(geo_flat)           # [B, 729, F]
        
        # 3. Fuse (Add & Norm)
        fused = self.layer_norm(sem_emb + geo_emb)
        
        # 4. Project to LLM
        out = self.output_proj(fused) # [B, 729, LLM_Dim]
        
        return out
