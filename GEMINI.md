# StarVLA Project Summary

## 1. Project Overview
**StarVLA** is a modular, "Lego-like" framework designed for developing Vision-Language-Action (VLA) models. Its primary goal is to enable rapid prototyping and research by decoupling the Vision-Language Model (VLM) backbone, action prediction heads, and data pipelines. This allows researchers to easily swap components (e.g., changing the VLM from Qwen to Florence-2, or the action head from Diffusion to L1 Regression) without rewriting the entire codebase.

## 2. Architecture (`starVLA/model`)
The core modeling logic resides in `starVLA/model`, built around a flexible hierarchy:

### Base Framework
- **`baseframework`** (`starVLA/model/framework/base_framework.py`): The parent class for all models. It handles:
    - **Checkpoint Loading**: Unified loading of config, weights, and normalization stats.
    - **Action Normalization**: Utilities to normalize/unnormalize actions (critical for robotics).
    - **Trainable Module Discovery**: Helper methods to identify which parts of the model should be frozen or trained.

### Model Frameworks
Different architectural patterns are implemented as "Frameworks" in `starVLA/model/framework/`:
- **`InternVLA-M1`** / **`QwenGR00T`**: Combines a **Qwen-VL** backbone with a **Flow-matching (DiT)** action head. 
    - **Flow:** VLM (Semantics) + DINO (Geometry) -> QFormer (Fusion) -> Diffusion Head (Action).
    - **Key Feature:** Extracts intermediate hidden states from Qwen for better robotic control.
- **`QwenDual`**: Similar to `QwenGR00T` but augments VLM features with **DINOv2** visual features for better spatial understanding.
- **`QwenAdapter`**: Injects learnable **"action queries"** into the VLM's input sequence. The output embeddings corresponding to these queries are projected to actions via an L1 regression head.
- **`QwenFAST`**: A discretization-based approach where actions are tokenized and predicted autoregressively by the VLM itself (treating actions as text/tokens).

### Components (The "Lego Bricks")
To add new models, you write **Wrappers** in `starVLA/model/modules/` that standardize their APIs:
- **VLM Backbones** (`starVLA/model/modules/vlm/`): Wrappers for foundation models like **Qwen2.5-VL**, **Qwen3-VL**, and **Florence-2**. They provide a consistent interface (`build_inputs`, `get_vlm_hidden_states`).
- **Action Heads** (`starVLA/model/modules/action_model/`): Modular heads including:
    - `FlowmatchingActionHead` / `DiTActionHeader`: Diffusion-based continuous action prediction.
    - `L1ActionHead`: Simple MLP for direct regression.

## 3. Training (`starVLA/training`)
Training is orchestrated via scripts in `starVLA/training/` using Hugging Face Accelerate. The system supports:
- **VLA Training** (`train_starvla.py`): Standard training of the VLA model on robotic datasets.
- **VLM Finetuning** (`train_starvlm.py`): Pure VLM instruction tuning.
- **Co-Training** (`train_starvla_cotrain.py`): Jointly training on VLM datasets (captioning/VQA) and Robotic datasets (actions) to maintain general capabilities.

**Key Features:**
- **Flexible Freezing**: A regex-based system allows fine-grained control over which modules (backbone, adapter, head) are frozen or trained.
- **Single Config**: The same YAML file used for training is also used to initialize the model architecture.

## 4. Data Pipeline (`starVLA/dataloader`)
The data layer relies heavily on **Hugging Face LeRobot** for standardization but adds a flexible wrapper:
- **`build_dataloader`**: The entry point that switches between Robotic data and VLM data.
- **Robotic Data (`gr00t_lerobot`)**:
    - **`LeRobotSingleDataset`**: Reads Parquet files for a specific robot.
    - **`LeRobotMixtureDataset`**: Mixes multiple datasets (e.g., Libero + Bridge) with weighted sampling.
    - **`data_config.py`**: Maps robot names to specific data schemas (keys for images/actions).
    - **Transforms**: Handles critical **normalization** of actions (using stats like `q01`, `q99`) and image resizing.
- **VLM Data (`vlm_datasets.py`)**: Handles standard vision-language datasets (e.g., ShareGPT4V), performing lazy loading and tokenization.

## 5. Usage & Workflows
- **Configuration**: Managed via **OmegaConf** and YAML files (e.g., `starvla_cotrain_libero.yaml`). This file acts as the universal blueprint for both the Trainer and the Model.
- **Execution**: Shell scripts in `examples/` act as the primary entry points.
    - Example: `examples/LIBERO/train_files/run_libero_train.sh` sets up the environment and calls the python training script.
- **Deployment**: `deployment/model_server/server_policy.py` wraps the trained model to serve requests via websockets, converting model outputs back to real-world actions using `unnormalize_actions`.

## 6. Extending the Framework
To add a custom pipeline (e.g., Llama-3 + GMM Head):
1.  **Wrappers**: Write `Llama3Wrapper` and `GMMHeadWrapper` in `model/modules/`.
2.  **Framework**: Write `MyCustomFramework.py` that connects them.
3.  **Registry**: Register it via `@FRAMEWORK_REGISTRY.register("MyName")`.
4.  **Config**: Create a YAML file pointing `framework.name` to `"MyName"` and setting the specific params for your new wrappers.

---
*Generated by Gemini CLI Agent on 2026-01-19*