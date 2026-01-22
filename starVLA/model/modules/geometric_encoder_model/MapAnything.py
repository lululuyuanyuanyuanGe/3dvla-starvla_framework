import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import numpy as np
import json
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import unittest.mock

class MapAnything(nn.Module):
    """
    Wrapper for the MapAnything geometric encoder model.
    Extracts 3D geometric features (metric 3D points) from 2D images.
    
    This implementation uses a 'Manual Offline Load' to skip redundant 
    torch.hub downloads from AWS/S3 that often fail in restricted clusters.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Handle cases where config might be a dict (for testing) or OmegaConf
        if isinstance(config, dict):
            model_config = config.get("framework", {}).get("geometric_encoder_model", {})
        else:
            model_config = config.framework.geometric_encoder_model
        
        self.model_id = getattr(model_config, "model_path", "facebook/map-anything")
        self.freeze = getattr(model_config, "freeze", True)
        self.image_size = getattr(model_config, "image_size", 518) 
        
        # Try to import MapAnything components
        try:
            from mapanything.models import MapAnything as MapAnythingModel
            from mapanything.utils.image import preprocess_inputs
            self.preprocess_inputs = preprocess_inputs
        except ImportError:
            raise ImportError(
                "MapAnything is not installed. Please install it using: "
                "pip install git+https://github.com/facebookresearch/map-anything.git"
            )
            
        print(f"--- [3DVLA] Starting Manual Offline Load for {self.model_id} ---")
        
        # 1. Download/Load config.json (This uses HF Mirror via HF_ENDPOINT)
        config_path = hf_hub_download(repo_id=self.model_id, filename="config.json")
        with open(config_path, "r") as f:
            map_config = json.load(f)
            
        # 2. HACK: Disable internet-based backbone downloads
        # We REMOVE 'pretrained' injection because uniception rejects it.
        # Instead, we rely entirely on the torch.hub patch below.
        
        # 3. Instantiate Model Structure (with random weights)
        # We filter the config dict to match MapAnything's __init__ signature
        init_args = map_config.copy()
        init_args.pop("transformers_version", None)
        init_args.pop("_name_or_path", None)
        init_args.pop("architectures", None)
        init_args.pop("model_type", None)

        # CRITICAL PATCH 1: Mock load_state_dict_from_url to return empty dict
        def no_download_side_effect(*args, **kwargs):
            print(f"Intercepted URL download request. Skipping download.")
            return {}

        # CRITICAL PATCH 2: Mock nn.Module.load_state_dict to ignore missing keys
        # The DINOv2 hub code calls load_state_dict(strict=True), which crashes with empty dict.
        # We force strict=False.
        original_load_state_dict = nn.Module.load_state_dict
        def loose_load_state_dict(self, state_dict, strict=True):
            # We only want to loose-load during this initialization phase
            return original_load_state_dict(self, state_dict, strict=False)

        print("Patching torch.hub and nn.Module.load_state_dict...")
        with unittest.mock.patch('torch.hub.load_state_dict_from_url', side_effect=no_download_side_effect), \
             unittest.mock.patch('torch.nn.Module.load_state_dict', side_effect=loose_load_state_dict, autospec=True):
            self.model = MapAnythingModel(**init_args)
        
        # 4. Load the real weights from safetensors (This uses HF Mirror)
        print("Downloading/Loading 4.91GB weights from HF Mirror...")
        weights_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        
        # Overwrite random weights with trained weights
        msg = self.model.load_state_dict(state_dict, strict=False)
        # This is only for testing
        print(f"Weights Loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        if len(msg.missing_keys) > 0:
            print(f"First 10 missing keys: {msg.missing_keys[:10]}")
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        print("--- [3DVLA] MapAnything Initialization Complete ---")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Batch of images [B, 3, H, W]. 
            
        Returns:
            pts3d: 3D geometric features [B, H, W, 3]
        """
        batch_size = images.shape[0]
        images_cpu = images.detach().cpu()
        
        # Convert range if needed [0, 1] -> [0, 255]
        if images_cpu.max() <= 1.0:
            images_cpu = images_cpu * 255.0
            
        views_example = []
        for i in range(batch_size):
            img_np = images_cpu[i].permute(1, 2, 0).numpy().astype(np.uint8)
            views_example.append({"img": img_np})
            
        processed_views = self.preprocess_inputs(
            views_example,
            resolution_set=self.image_size,
            norm_type="dinov2"
        )
        
        with torch.no_grad():
             predictions = self.model.infer(
                processed_views,
                memory_efficient_inference=True, 
                apply_mask=True,
                apply_confidence_mask=False
            )
             
        batch_pts3d = []
        for pred in predictions:
            pts = pred["pts3d"]
            # Convert numpy to torch
            if isinstance(pts, np.ndarray):
                pts = torch.from_numpy(pts)
            if pts.ndim == 4:
                pts = pts.squeeze(0)
            batch_pts3d.append(pts)
            
        return torch.stack(batch_pts3d).to(images.device)

    def get_output_dim(self):
        return 3

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from PIL import Image
    
    mock_config = {
        "framework": {
            "geometric_encoder_model": {
                "model_path": "facebook/map-anything",
                "freeze": True,
                "image_size": 518
            }
        }
    }
    
    cfg = OmegaConf.create(mock_config)
    
    try:
        print("Initializing MapAnything Wrapper...")
        model = MapAnything(cfg)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("Moved model to CUDA")
        
        print("Creating dummy input...")
        input_tensor = torch.rand(1, 3, 518, 518)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        print("Running Inference...")
        output = model(input_tensor)
        
        print(f"Output Shape: {output.shape}")
        print(f"Output tensors: {output}")
        print("Success!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
