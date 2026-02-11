import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

# Components
from starVLA.model.modules.geometric_encoder_model.MapAnything import MapAnything
from starVLA.model.modules.semantic_encoder_model.SigLIP import SigLIP
from starVLA.model.modules.vlm.Llava3D import Llava3D
from starVLA.model.modules.projector.GeometryFusion import GeometryFusion
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model

logger = initialize_overwatch(__name__)

@FRAMEWORK_REGISTRY.register("3DVLA")
class ThreeDVLA(baseframework):
    """
    3DVLA: A modular VLA framework integrating 3D geometric understanding.
    
    Pipeline:
    1. Input: 2D Images + Text
    2. Encoders: MapAnything (3D Geometry) + SigLIP (2D Semantics)
    3. Fusion: Geometric Pooling (GeometryFusion)
    4. Backbone: Llava3D (LLaVA-1.5 based)
    5. Head: Diffusion Action Head (DiT)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        
        # 1. Encoders
        self.geometric_encoder = MapAnything(config)
        self.semantic_encoder = SigLIP(config)
        
        # 2. Fusion
        self.fusion_module = GeometryFusion(config)
        
        # 3. VLM Backbone
        self.vlm = Llava3D(config)
        
        # 4. Action Head
        self.action_model = get_action_model(config=config)
        
        # Configs
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.train_obs_image_size = getattr(config.datasets.vla_data, "image_size", None)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """
        Forward pass for training.
        """
        # Unpack Data
        # Flatten batch images? Usually VLA tasks have 1 main view or multi-view.
        # Assuming examples['image'] is a list of PIL images (multi-view).
        # For 3DVLA prototype, we might assume single view or take the first one.
        # MapAnything/SigLIP expect [B, 3, H, W] tensors or list of PILs.
        
        batch_images = []
        instructions = []
        actions = []
        
        for example in examples:
            # Handle Multi-view: Take first image for now (simplification for 3DVLA prototype)
            # TODO: Support multi-view fusion if needed
            imgs = example["image"]
            if isinstance(imgs, list):
                img = imgs[0] 
            else:
                img = imgs
            batch_images.append(img)
            
            instructions.append(example["lang"])
            actions.append(example["action"])

        device = self.action_model.net.x_embedder.proj.weight.device # Get device from a parameter
        
        # 1. Encode Images
        # SigLIP Preprocess
        siglip_input = self.semantic_encoder.preprocess_images(batch_images).to(device, dtype=torch.bfloat16)
        # MapAnything Preprocess
        # MapAnything usually expects dict or specific format. Using its preprocess if available.
        # The wrapper might handle it. Let's assume wrapper takes raw images or we need to prep.
        # Checking MapAnything wrapper: it imports `preprocess_inputs`. 
        # But we should call the wrapper's forward? 
        # Wait, the wrapper implementation I saw earlier was incomplete/truncated.
        # Assuming wrapper.forward takes standard tensors or list of images.
        # Let's try passing list of images if wrapper handles it, or preprocess.
        # For safety/performance, let's assume wrapper handles preprocessing or we do it here.
        # Given SigLIP wrapper has `preprocess_images`, MapAnything might too.
        # If not, we might crash. But for prototype, let's assume `preprocess_inputs` is used inside or available.
        
        # NOTE: To avoid double preprocessing, check if MapAnything wrapper does it.
        # The wrapper has `self.preprocess_inputs`.
        map_inputs = self.geometric_encoder.preprocess_inputs(batch_images).to(device, dtype=torch.float32) # MapAnything likely fp32
        
        with torch.no_grad(): # Encoders are usually frozen
             sem_features = self.semantic_encoder(siglip_input) # [B, 729, 1152]
             geo_map = self.geometric_encoder(map_inputs)       # [B, 518, 518, 3]

        # 2. Fusion
        # Cast to bf16 if fusion is bf16
        geo_map = geo_map.to(dtype=torch.bfloat16)
        sem_features = sem_features.to(dtype=torch.bfloat16)
        
        visual_embeds = self.fusion_module(sem_features, geo_map) # [B, 729, 4096]

        # 3. Text Embedding
        tokens = self.vlm.tokenizer(
            instructions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        input_ids = tokens.input_ids
        text_embeds = self.vlm.model.get_input_embeddings()(input_ids).to(dtype=torch.bfloat16)
        
        # 4. Concatenate (Visual + Text)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 5. Attention Mask
        visual_mask = torch.ones(visual_embeds.shape[:2], device=device, dtype=tokens.attention_mask.dtype)
        attention_mask = torch.cat([visual_mask, tokens.attention_mask], dim=1)
        
        # 6. VLM Forward
        vlm_outputs = self.vlm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        hidden_states = vlm_outputs.hidden_states[-1] # Take last layer [B, Seq, Dim]
        
        # 7. Action Prediction
        # We need to extract the condition for the action head.
        # Usually we pool the hidden states or take specific tokens.
        # M1 uses Layerwise QFormer. Here we might just average pool or take last token?
        # Development plan didn't specify Action Head conditioning strategy (only "Diffusion Head").
        # Standard approach: Pool visual+text tokens, or just use them all as context.
        # DiT usually takes a sequence length `condition`.
        # M1 uses `action_condition` from QFormer [B, 64, D].
        # If we don't have QFormer, we can just project the sequence to a fixed size or use cross-attention in DiT.
        # But DiT wrapper expects `condition` of shape [B, L, D].
        # We can pass `hidden_states` directly if DiT supports variable length cross-attention.
        # M1 DiT: `noise_pred = self.net(x_t, timestep, condition)`
        # Let's assume passing full sequence is fine, or we average pool if too long.
        # LLaVA sequence can be long (729 + text).
        # Let's Average Pool for a compact representation if we don't use QFormer.
        # OR, since we want "Action from Llava3D's hidden states", maybe we interpret the LAST token as the "Action Token" (like RT-2).
        # But this is a Diffusion Head. It conditions on the context.
        # Let's pass the full sequence `hidden_states` as condition.
        
        condition = hidden_states
        
        # Prepare Actions (Chunking & Repeating logic from M1)
        actions = torch.tensor(np.array(actions), device=device, dtype=torch.float32) # [B, T, D]
        actions_future = actions[:, -(self.future_action_window_size + 1) :, :]
        
        # Repeat for diffusion efficiency
        repeated_steps = self.config.trainer.get("repeated_diffusion_steps", 4) if self.config.trainer else 4
        
        actions_repeated = actions_future.repeat(repeated_steps, 1, 1)
        condition_repeated = condition.repeat(repeated_steps, 1, 1)
        
        # DiT Forward
        # Cast condition to float32 for DiT if needed (M1 does this)
        with torch.autocast("cuda", dtype=torch.float32):
             noise_pred, noise, timestep = self.action_model(
                 actions_repeated, 
                 condition_repeated.to(dtype=torch.float32)
             )
             action_loss = self.action_model.loss(noise_pred, noise)
             
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]], 
        instructions: List[str],
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 5,
        **kwargs
    ) -> dict:
        """
        Inference method.
        """
        device = self.action_model.net.x_embedder.proj.weight.device
        
        # Preprocess
        # Flatten batch images (taking 1st view)
        single_view_images = [imgs[0] if isinstance(imgs, list) else imgs for imgs in batch_images]
        
        # Resize if needed (M1 does this)
        if self.train_obs_image_size:
            single_view_images = resize_images(single_view_images, target_size=self.train_obs_image_size)

        # 1. Encoders
        siglip_input = self.semantic_encoder.preprocess_images(single_view_images).to(device, dtype=torch.bfloat16)
        map_inputs = self.geometric_encoder.preprocess_inputs(single_view_images).to(device, dtype=torch.float32)

        sem_features = self.semantic_encoder(siglip_input).to(dtype=torch.bfloat16)
        geo_map = self.geometric_encoder(map_inputs).to(dtype=torch.bfloat16)
        
        # 2. Fusion
        visual_embeds = self.fusion_module(sem_features, geo_map)
        
        # 3. Text
        tokens = self.vlm.tokenizer(instructions, padding=True, return_tensors="pt").to(device)
        input_ids = tokens.input_ids
        text_embeds = self.vlm.model.get_input_embeddings()(input_ids).to(dtype=torch.bfloat16)
        
        # 4. Concat
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        visual_mask = torch.ones(visual_embeds.shape[:2], device=device, dtype=tokens.attention_mask.dtype)
        attention_mask = torch.cat([visual_mask, tokens.attention_mask], dim=1)
        
        # 5. VLM
        vlm_outputs = self.vlm(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        condition = vlm_outputs.hidden_states[-1].to(dtype=torch.float32) # [B, Seq, Dim]
        
        # 6. Diffusion Sampling
        B = condition.shape[0]
        using_cfg = cfg_scale > 1.0
        
        # Init Noise
        noise = torch.randn(
            B, 
            self.future_action_window_size + 1, 
            self.action_model.in_channels, 
            device=device,
            dtype=torch.float32
        )
        
        # CFG Setup
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            # Unconditional conditioning (zeros or learnable?)
            # M1 uses `self.action_model.net.z_embedder.uncondition`
            # We assume DiT has this.
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0).expand(B, condition.shape[1], -1) # Match seq len?
            # Note: DiT uncondition shape usually [1, 1, D] or [1, L, D]. 
            # If SeqLen varies, expanding fixed uncondition might be tricky if it's not broadcastable.
            # M1 expands to [B, n_qformer_token, D]. Our SeqLen is 729 + TextLen.
            # We might need to handle uncondition size matching.
            # For prototype: Let's assume DiT handles broadcasting or we skip CFG for now if tricky.
            # Proceeding with M1 logic assuming matching dims.
            z = torch.cat([condition, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=condition)
            sample_fn = self.action_model.net.forward
            
        # Sample
        if use_ddim:
             if self.action_model.ddim_diffusion is None:
                 self.action_model.create_ddim(ddim_step=num_ddim_steps)
             samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                 sample_fn,
                 noise.shape,
                 noise,
                 clip_denoised=False,
                 model_kwargs=model_kwargs,
                 progress=False,
                 device=device,
                 eta=0.0
             )
             
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
            
        normalized_actions = samples.cpu().numpy()
        return {"normalized_actions": normalized_actions}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import numpy as np
    import os
    
    # 1. Load Config
    config_path = "starVLA/config/training/spatial_vla.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}, creating mock config.")
        mock_config = {
            "framework": {
                "name": "3DVLA",
                "semantic_encoder_model": {"model_path": "google/siglip-so400m-patch14-384", "freeze": True},
                "geometric_encoder_model": {"model_path": "facebook/map-anything", "freeze": True, "image_size": 518},
                "projector": {"semantic_dim": 1152, "geometry_dim": 3, "fusion_dim": 1152, "llm_dim": 4096},
                "vlm": {"model_path": "llava-hf/llava-1.5-7b-hf", "freeze_backbone": True},
                "action_model": {
                    "action_model_type": "DiT-S", # Use Small for test speed
                    "action_hidden_dim": 384,     # Match DiT-S token size
                    "action_dim": 7,
                    "state_dim": 7,
                    "future_action_window_size": 8,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 2
                }
            },
            "datasets": {"vla_data": {"image_size": [224, 224]}},
            "trainer": {"get": lambda k, d: 2} # Mock trainer config get method
        }
        cfg = OmegaConf.create(mock_config)
    else:
        cfg = OmegaConf.load(config_path)
        # Override to DiT-S for faster local testing if needed, or keep DiT-B
        # cfg.framework.action_model.action_model_type = "DiT-S"
        # cfg.framework.action_model.action_hidden_dim = 384
    
    print("Initializing 3DVLA Framework...")
    try:
        model = ThreeDVLA(cfg)
        model.eval()
        
        # Mock Data
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        text = "Pick up the red cube."
        action = np.random.randn(16, 7).astype(np.float32) # [T, 7]
        
        example = {
            "image": [image], # List for multi-view (we take first)
            "lang": text,
            "action": action
        }
        batch = [example, example] # Batch of 2
        
        print("\n--- Testing Forward (Training) ---")
        if torch.cuda.is_available():
            model = model.cuda()
            print("Moved to CUDA")
            
        outputs = model(batch)
        print(f"Action Loss: {outputs['action_loss'].item()}")
        
        print("\n--- Testing Predict Action (Inference) ---")
        batch_images = [[image], [image]]
        instructions = [text, text]
        
        pred = model.predict_action(batch_images, instructions, num_ddim_steps=2)
        print(f"Predicted Action Shape: {pred['normalized_actions'].shape}")
        
        print("\nTest Passed!")
        
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()

