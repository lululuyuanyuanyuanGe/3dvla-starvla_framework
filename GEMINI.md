# StarVLA Project Summary

## 1. Project Overview
**StarVLA** is a modular, "Lego-like" framework designed for developing Vision-Language-Action (VLA) models. Its primary goal is to enable rapid prototyping and research by decoupling the Vision-Language Model (VLM) backbone, action prediction heads, and data pipelines. This allows researchers to easily swap components (e.g., changing the VLM from Qwen to Florence-2, or the action head from Diffusion to L1 Regression) without rewriting the entire codebase.

## 2. Architecture (`starVLA/model`)
The core modeling logic resides in `starVLA/model`, built around a flexible hierarchy:

### Base Framework
- **`baseframework`** (`starVLA/model/framework/base_framework.py`): The parent class for all models. It handles checkpoint loading, action normalization, and trainable module discovery.

### Model Frameworks
- **`InternVLA-M1`** / **`QwenGR00T`**: Qwen-VL + DINO + QFormer + Diffusion Head.
- **`QwenDual`**: Augments VLM with DINOv2 for better spatial understanding.
- **`QwenAdapter`**: Injects learnable action queries into the VLM sequence.
- **`QwenFAST`**: Tokenizes actions for autoregressive prediction by the VLM.
- **`3DVLA` (SpatialVLA)**: A new architecture explicitly designed for 3D spatial grounding.
    - **Pipeline:** MapAnything (Geometry) + SigLIP (Semantics) -> Fusion -> Llava3D (Brain) -> Diffusion Head (Action).

### Components (The "Lego Bricks")
- **VLM Backbones** (`starVLA/model/modules/vlm/`): Wrappers for Qwen2.5-VL, Llava3D, etc.
- **Action Heads** (`starVLA/model/modules/action_model/`): Diffusion-based (DiT/Flowmatching) or L1 MLP heads.
- **Geometric Encoders** (`starVLA/model/modules/geometric_encoder_model/`):
    - **MapAnything**: A universal metric 3D reconstruction model.
        - **Capabilities**: Can infer camera intrinsics, poses, and global metric scale from images alone (zero-shot).
        - **Representation**: Predicts a factored scene representation (ray maps, depth maps, scale factor).
        - **Output**: Generates globally consistent per-view point maps ($X \in \mathbb{R}^{B \times H \times W \times 3}$) where each pixel is mapped to metric $(x, y, z)$ coordinates.
- **Semantic Encoders** (`starVLA/model/modules/semantic_encoder_model/`):
    - **SigLIP**: Extracts high-level 2D semantic features to complement 3D geometry.

## 3. Training (`starVLA/training`)
Orchestrated via Hugging Face Accelerate + DeepSpeed. Supports standard VLA training, VLM finetuning, and Co-Training (Robotics + Language).

## 4. Data Pipeline (`starVLA/dataloader`)
Relies on **Hugging Face LeRobot** format.
- **`LeRobotMixtureDataset`**: Handles multi-robot dataset blending.
- **`data_config.py`**: Maps embodiment-specific keys.
- **Spatial Data**: 3DVLA can leverage MapAnything to handle datasets without camera priors (like LIBERO) by reconstructing the 3D scene from RGB inputs.

## 5. Usage & Workflows
- **Configuration**: A single YAML file (e.g., `spatial_vla.yaml`) serves as the universal blueprint for both the model architecture and the trainer.
- **Deployment**: Policies are served via websockets using `server_policy.py`, converting normalized outputs back to robot-specific units.

---
*Updated by Gemini CLI Agent on 2026-01-21*
