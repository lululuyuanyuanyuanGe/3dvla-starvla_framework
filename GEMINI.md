# StarVLA Project Summary

## 1. Project Overview
**StarVLA** is a modular, "Lego-like" framework designed for developing Vision-Language-Action (VLA) models. Its primary goal is to enable rapid prototyping and research by decoupling the Vision-Language Model (VLM) backbone, action prediction heads, and data pipelines.

## 2. Architecture (`starVLA/model`)
The core modeling logic resides in `starVLA/model`, built around a flexible hierarchy:

### Base Framework
- **`baseframework`** (`starVLA/model/framework/base_framework.py`): The parent class for all models. It handles checkpoint loading, action normalization, and trainable module discovery.

### Model Frameworks
- **`InternVLA-M1`** / **`QwenGR00T`**: Qwen-VL + DINO + QFormer + Diffusion Head.
- **`QwenDual`**: Augments VLM with DINOv2 for better spatial understanding.
- **`3DVLA` (SpatialVLA)**: A new architecture explicitly designed for 3D spatial grounding.
    - **Pipeline:** MapAnything (Geometry) + SigLIP (Semantics) -> Fusion -> Llava3D (Brain) -> Diffusion Head (Action).

### Components (The "Lego Bricks")
- **VLM Backbones** (`starVLA/model/modules/vlm/`): Wrappers for Qwen2.5-VL, Llava3D, etc.
- **Action Heads** (`starVLA/model/modules/action_model/`): Diffusion-based (DiT/Flowmatching) or L1 MLP heads.
- **Geometric Encoders** (`starVLA/model/modules/geometric_encoder_model/`):
    - **MapAnything**: A universal metric 3D reconstruction model.
        - **Capabilities**: Zero-shot inference of camera intrinsics, poses, and global metric scale.
        - **Output**: Globally consistent per-view point maps `[B, 518, 518, 3]` mapping pixels to metric $(x, y, z)$.
- **Semantic Encoders** (`starVLA/model/modules/semantic_encoder_model/`):
    - **SigLIP**: Extracts high-level 2D semantic features.
        - **Model**: `google/siglip-so400m-patch14-384`.
        - **Output**: Dense tokens `[B, 729, 1152]` (27x27 grid).

## 3. Training (`starVLA/training`)
Orchestrated via Hugging Face Accelerate + DeepSpeed. Supports standard VLA training, VLM finetuning, and Co-Training.

## 4. Data Pipeline (`starVLA/dataloader`)
Relies on **Hugging Face LeRobot** format. For 3DVLA, the pipeline must handle dual-resolution/dual-normalization inputs for MapAnything (518px, DINO) and SigLIP (384px, SigLIP).

---
*Updated by Gemini CLI Agent on 2026-01-21*