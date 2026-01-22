# Development Plan: 3DVLA

## 1. Objective
Implement **3DVLA**, a new modular Vision-Language-Action framework that explicitly incorporates 3D geometric understanding.

**Architecture Flow:**
1.  **Input:** 2D Images + Text Instructions.
2.  **Geometry Encoder:** **MapAnything** (reconstructs/extracts 3D geometric features from 2D).
3.  **Semantic Encoder:** **SigLIP** (extracts high-level 2D semantic features).
4.  **Fusion Module:** Fuses 3D geometry + 2D semantics into a unified representation.
5.  **3D VLM Backbone:** **Llava3D** (processes the fused representation with 3D awareness).
6.  **Action Head:** **Diffusion Head** (predicts robot actions from Llava3D's hidden states).

## 2. Directory Structure Changes
We will create new directories to enforce a clean separation of concerns, moving away from the generic `dino_model` folder.

```text
starVLA/model/modules/
├── semantic_encoder_model/  <-- NEW (for SigLIP)
├── geometric_encoder_model/ <-- NEW (for MapAnything)
├── vlm/                     <-- Existing (Add Llava3D here)
├── projector/               <-- Existing (Add Fusion module here)
└── action_model/            <-- Existing (Reuse Diffusion head)
```

## 3. Implementation Steps

### Phase 1: Infrastructure & Scaffolding (DONE)
1.  **Create Directories:** Set up the new folders in `starVLA/model/modules/`. (DONE)
2.  **Update `__init__.py` files:** Ensure the new folders and modules are importable and reachable by the registry. (DONE)

### Phase 2: Component Implementation (The Wrappers)
Each wrapper must implement the standard interface (config-based init, `forward` returning features).

1.  **Geometric Encoder (`MapAnything`):** (DONE)
    *   **File:** `starVLA/model/modules/geometric_encoder_model/MapAnything.py`
    *   **Status:** Implemented and Verified.
    *   **Verification:** `python starVLA/model/modules/geometric_encoder_model/MapAnything.py` successfully outputs globally consistent point maps `[B, 518, 518, 3]`.

2.  **Semantic Encoder (`SigLIP`):** (DONE)
    *   **File:** `starVLA/model/modules/semantic_encoder_model/SigLIP.py`
    *   **Status:** Implemented and Verified.
    *   **Task:** Wrap the Hugging Face SigLIP model to extract 2D semantic features.
    *   **Input:** 2D Images (standardized to 384x384).
    *   **Output:** Dense semantic feature maps `[B, 729, 1152]`.
        *   **Tokens (729):** Derived from 27x27 grid patches (384px / patch size 14).
        *   **Dimension (1152):** Hidden embedding size for `so400m` variant.
    *   **Verification:** Successful inference on dummy input with expected tensor values.

3.  **3D VLM (`Llava3D`):**
    *   **File:** `starVLA/model/modules/vlm/Llava3D.py`
    *   **Strategy:** Custom implementation using standard LLaVA backbone + Custom 3D Logic.
    *   **Rationale:** Avoids dependency hell of `LLaVA-3D` repo and bypasses its redundant internal depth estimation.
    *   **Task:** Wrap standard LLaVA model. Implement `forward` to accept *pre-fused* embeddings instead of raw images.

### Phase 3: The Glue (Fusion & Framework)

1.  **Fusion Module:**
    *   **File:** `starVLA/model/modules/projector/GeometryFusion.py`
    *   **Alignment Strategy:** **Geometric Pooling**.
        *   MapAnything Output: `[B, 518, 518, 3]` (Dense Points)
        *   SigLIP Output: `[B, 729, 1152]` (27x27 Patches)
        *   **Action:** Resize/Pool MapAnything to `[B, 27, 27, 3]` using `AdaptiveAvgPool2d`. Flatten to `[B, 729, 3]`.
    *   **Fusion Logic:**
        *   Project SigLIP: `Linear(1152 -> 1152)`
        *   Embed Geometry: `MLP(3 -> 1152)` (Positional Encoding)
        *   Add: `Fused = Semantics + Geometry_Embedding`
        *   Project to LLM: `Linear(1152 -> 4096)`

2.  **Framework Class (`3DVLA`):**
    *   **File:** `starVLA/model/framework/3DVLA.py`
    *   **Task:** The master class inheriting from `baseframework`.
    *   **Logic:**
        *   `__init__`: Load all 4 components (Geometry, Semantic, VLM, Action).
        *   `forward`: Chain them together: `Img -> (Geo, Sem) -> Fusion -> VLM -> ActionHead -> Loss`.
        *   `predict_action`: Same chain, but sampling from the ActionHead.
    *   **Registration:** Decorate with `@FRAMEWORK_REGISTRY.register("3DVLA")`.

### Phase 4: Configuration & Data
1.  **Config File:** Create `starVLA/config/training/spatial_vla.yaml`.
    *   Define the new structure (`framework.geometric_encoder`, `framework.semantic_encoder`, etc.).
2.  **Data Config (if needed):** Verify if `starVLA/dataloader/gr00t_lerobot/data_config.py` needs updates to load any special data required by MapAnything (e.g., camera intrinsics).

## 4. Execution Plan
*   **Step 1:** Verify MapAnything and SigLIP wrappers. (DONE)
*   **Step 2:** Implement Llava3D wrapper (Custom).
*   **Step 3:** Implement Fusion and Framework.
*   **Step 4:** Create Config and Dry Run.
