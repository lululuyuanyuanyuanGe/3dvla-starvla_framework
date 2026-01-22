import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import numpy as np

class MapAnything(nn.Module):
    """
    Wrapper for the MapAnything geometric encoder model.
    Extracts 3D geometric features (metric 3D points) from 2D images.
    
    See: https://github.com/facebookresearch/map-anything
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
        
        # Try to import MapAnything
        try:
            from mapanything.models import MapAnything as MapAnythingModel
            from mapanything.utils.image import preprocess_inputs
            self.preprocess_inputs = preprocess_inputs
        except ImportError:
            raise ImportError(
                "MapAnything is not installed. Please install it using: "
                "pip install git+https://github.com/facebookresearch/map-anything.git"
            )
            
        # Load the model
        print(f"Loading MapAnything from {self.model_id}...")
        self.model = MapAnythingModel.from_pretrained(self.model_id)
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Batch of images [B, 3, H, W]. 
                    IMPORTANT: MapAnything expects images in [0, 255] range for preprocessing.
                    If your dataloader normalizes to [0, 1], we multiply by 255.
            
        Returns:
            pts3d: 3D geometric features [B, H, W, 3]
        """
        batch_size = images.shape[0]
        
        # Ensure images are on CPU for preprocessing (MapAnything utils usually expect CPU/Numpy)
        # and in range [0, 255] as expected by preprocess_inputs
        images_cpu = images.detach().cpu()
        if images_cpu.max() <= 1.0:
            images_cpu = images_cpu * 255.0
            
        views_example = []
        for i in range(batch_size):
            # Convert [3, H, W] -> [H, W, 3] for MapAnything
            img_np = images_cpu[i].permute(1, 2, 0).numpy().astype(np.uint8)
            views_example.append({"img": img_np})
            
        # Use MapAnything's own preprocessing
        processed_views = self.preprocess_inputs(
            views_example,
            resolution_set=self.image_size,
            norm_type="dinov2" # MapAnything default
        )
        
        # Run inference
        with torch.no_grad(): # usually we don't train MapAnything
             predictions = self.model.infer(
                processed_views,
                memory_efficient_inference=True, 
                apply_mask=True,
                apply_confidence_mask=False
            )
             
        # Extract the 3D points
        batch_pts3d = []
        for pred in predictions:
            # pred["pts3d"] is [B, H, W, 3] usually, but let's be safe
            pts = pred["pts3d"]
            if pts.ndim == 4:
                pts = pts.squeeze(0) # Ensure [H, W, 3]
            batch_pts3d.append(pts)
            
        # Stack back into a batch tensor
        return torch.stack(batch_pts3d).to(images.device)

    def get_output_dim(self):
        # MapAnything outputs 3 coordinates (x, y, z) per pixel
        return 3

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from PIL import Image
    
    # Mock Config
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
        
        # Create a dummy image tensor [1, 3, 518, 518] in range [0, 1]
        print("Creating dummy input...")
        input_tensor = torch.rand(1, 3, 518, 518)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        print(f"Input Tensor Shape: {input_tensor.shape}")
        
        # Forward Pass
        print("Running Inference...")
        output = model(input_tensor)
        
        print(f"Output Shape: {output.shape}") # Should be [1, 518, 518, 3]
        print("Success!")
        
    except ImportError as e:
        print(f"Test Skipped: {e}")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()