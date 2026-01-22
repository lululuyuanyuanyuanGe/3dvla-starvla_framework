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

### Phase 1: Infrastructure & Scaffolding (IN PROGRESS)
1.  **Create Directories:** Set up the new folders in `starVLA/model/modules/`. (DONE)
2.  **Update `__init__.py` files:** Ensure the new folders and modules are importable and reachable by the registry. (DONE)

### Phase 2: Component Implementation (The Wrappers)
Each wrapper must implement the standard interface (config-based init, `forward` returning features).

1.  **Geometric Encoder (`MapAnything`):** (DONE)
    *   **File:** `starVLA/model/modules/geometric_encoder_model/MapAnything.py`
    *   **Status:** Implemented. Handles complex initialization for restricted network environments.
    *   **Network Bypass Strategy (Fixing `torch.hub` crashes):**
        *   **Problem:** The compute cluster blocks AWS/S3 (used by `torch.hub`), preventing DINOv2 weights download. `HF_ENDPOINT` only fixes Hugging Face, not `torch.hub`.
        *   **Solution:** A "Manual Offline Load" strategy with double patching.
            1.  **Mock URL Download:** Patched `torch.hub.load_state_dict_from_url` to return an empty dict `{}` instead of downloading.
            2.  **Mock Strict Loading:** Patched `nn.Module.load_state_dict` to force `strict=False` during initialization. This allows DINOv2 to initialize with the empty dict (random weights) instead of crashing on missing keys.
            3.  **Real Weight Load:** Manually downloaded `model.safetensors` (4.91GB) via HF Mirror (which works) and overwrote the random weights.
    *   **Verification:** `python starVLA/model/modules/geometric_encoder_model/MapAnything.py` runs successfully.

2.  **Semantic Encoder (`SigLIP`):**
    *   **File:** `starVLA/model/modules/semantic_encoder_model/SigLIP.py`
    *   **Task:** Wrap the Hugging Face SigLIP model.
    *   **Input:** 2D Images.
    *   **Output:** Dense 2D semantic feature maps.

3.  **3D VLM (`Llava3D`):**
    *   **File:** `starVLA/model/modules/vlm/Llava3D.py`
    *   **Task:** Wrap the Llava3D model.
    *   **Crucial:** Implement `build_inputs` to handle the custom fused data format and ensure `output_hidden_states=True` is supported.

### Phase 3: The Glue (Fusion & Framework)

1.  **Fusion Module:**
    *   **File:** `starVLA/model/modules/projector/GeometryFusion.py`
    *   **Task:** Implement a learnable module (e.g., Cross-Attention or MLP) to align and merge the output of MapAnything and SigLIP before passing to the VLM.

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
*   **Step 1:** Run `python starVLA/model/modules/geometric_encoder_model/MapAnything.py` to test MapAnything wrapper. (DONE)
*   **Step 2:** Implement wrappers (SigLIP first, then Llava3D).
*   **Step 3:** Implement Fusion and Framework.
*   **Step 4:** Create Config and Dry Run (verify dimensions match).
