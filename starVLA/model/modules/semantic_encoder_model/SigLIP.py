import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from PIL import Image
from transformers import AutoModel, AutoProcessor

class SigLIP(nn.Module):
    """
    Wrapper for the SigLIP semantic encoder model.
    Extracts high-level semantic features from 2D images.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if isinstance(config, dict):
            model_config = config.get("framework", {}).get("semantic_encoder_model", {})
        else:
            model_config = config.framework.semantic_encoder_model
            
        self.model_id = getattr(model_config, "model_path", "google/siglip-so400m-patch14-384")
        self.freeze = getattr(model_config, "freeze", True)
        
        print(f"Loading SigLIP from {self.model_id}...")
        
        # Load Processor and Vision Model
        # This respects HF_ENDPOINT for mirrors.
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # We only need the vision part of SigLIP for our VLA
        self.model = AutoModel.from_pretrained(self.model_id).vision_model
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess PIL images to tensors expected by SigLIP.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Batch of images [B, 3, H, W], ALREADY preprocessed by SigLIP processor.
            
        Returns:
            semantic_features: [B, N_patches, D] - dense semantic features
        """
        # SigLIP vision_model returns a dictionary
        # We want the patch embeddings (last_hidden_state)
        # Shape: [B, num_patches, hidden_size]
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state

    def get_output_dim(self):
        return self.model.config.hidden_size

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    # Mock Config
    mock_config = {
        "framework": {
            "semantic_encoder_model": {
                "model_path": "google/siglip-so400m-patch14-384",
                "freeze": True
            }
        }
    }
    
    cfg = OmegaConf.create(mock_config)
    
    try:
        print("Initializing SigLIP Wrapper...")
        model = SigLIP(cfg)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("Moved model to CUDA")
            
        # Create a dummy image (Blue Square)
        print("Creating dummy input...")
        dummy_img = Image.new('RGB', (384, 384), color='blue')
        
        # Preprocess
        input_tensor = model.preprocess_images([dummy_img])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        print(f"Input Tensor Shape: {input_tensor.shape}")
        
        # Forward Pass
        print("Running Inference...")
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output Shape: {output.shape}") # Expected [1, 729, 1152] for so400m-384
        print("Success!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
