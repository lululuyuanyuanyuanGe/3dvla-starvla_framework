import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, LlavaForConditionalGeneration
from typing import Optional, List, Dict, Any

class Llava3D(nn.Module):
    """
    Wrapper for the Llava3D (custom 3DVLA version) VLM.
    It uses a standard LLaVA backbone but accepts pre-fused 3D-aware embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if isinstance(config, dict):
            model_config = config.get("framework", {}).get("vlm", {})
        else:
            model_config = config.framework.vlm
            
        self.model_path = getattr(model_config, "model_path", "llava-hf/llava-1.5-7b-hf")
        self.freeze_backbone = getattr(model_config, "freeze_backbone", True)
        
        print(f"Loading LLaVA Backbone from {self.model_path}...")
        
        # Load Processor (Tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.tokenizer.padding_side = "left"
        
        # Load Model
        # We use LlavaForConditionalGeneration from Transformers
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto" # Let HF handle device placement
        )
        
        if self.freeze_backbone:
            print("Freezing LLaVA Backbone...")
            for param in self.model.parameters():
                param.requires_grad = False
                
    def get_vlm_hidden_states(self, 
                              input_ids: torch.LongTensor, 
                              inputs_embeds: torch.FloatTensor,
                              attention_mask: Optional[torch.LongTensor] = None):
        """
        Forward pass to get hidden states from the VLM.
        
        Args:
            input_ids: [B, Seq_Len] - Text tokens (instructions + special tokens)
            inputs_embeds: [B, Seq_Len, Hidden_Dim] - Fused Visual + Text embeddings.
                           NOTE: In standard HF LLaVA, you typically pass input_ids OR inputs_embeds.
                           If we pass inputs_embeds, it must contain EVERYTHING (images + text).
            attention_mask: [B, Seq_Len]
            
        Returns:
            hidden_states: [B, Seq_Len, Hidden_Dim] (from the last layer, or all layers)
        """
        # We need output_hidden_states=True to feed the Action Head
        outputs = self.model(
            input_ids=None, # We use inputs_embeds
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        return outputs

    def build_inputs(self, images, text):
        """
        Placeholder. The complex fusion logic (SigLIP + MapAnything) happens 
        OUTSIDE this wrapper in the Framework class (3DVLA.py) or Fusion Module.
        This wrapper mainly serves as the LLM interface.
        """
        raise NotImplementedError("Use 3DVLA Framework to build fused inputs.")

    def forward(self, input_ids, inputs_embeds, attention_mask=None):
        return self.get_vlm_hidden_states(input_ids, inputs_embeds, attention_mask)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    mock_config = {
        "framework": {
            "vlm": {
                "model_path": "llava-hf/llava-1.5-7b-hf",
                "freeze_backbone": True
            }
        }
    }
    cfg = OmegaConf.create(mock_config)
    
    try:
        print("Initializing Llava3D Wrapper...")
        model = Llava3D(cfg)
        
        print("Creating dummy inputs...")
        # LLaVA 7B hidden dim is 4096
        B, Seq_Len, Dim = 1, 10, 4096
        inputs_embeds = torch.randn(B, Seq_Len, Dim, dtype=torch.bfloat16).cuda()
        attention_mask = torch.ones(B, Seq_Len, dtype=torch.long).cuda()
        
        print("Running Inference...")
        outputs = model(None, inputs_embeds, attention_mask)
        
        print(f"Output Hidden States Shape: {outputs.hidden_states[-1].shape}")
        print("Success!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
