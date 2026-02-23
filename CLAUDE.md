# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

StarVLA is a modular, "Lego-like" codebase for developing Vision-Language-Action (VLA) models. The key design principle is **high cohesion, low coupling**: each component (framework, VLM, action head, dataloader) is independently testable and swappable.

This fork extends upstream StarVLA with a custom spatial VLM called **MapAnythingLlava3D** — a model that fuses 2D vision, 3D geometric features, and language for spatially-aware action prediction.

## Environment Setup

```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # must match CUDA toolkit + PyTorch
pip install -e .
```

Verify flash-attn compatibility: `nvcc -V && pip list | grep -E 'torch|flash-attn'`

## Common Commands

### Build / Lint / Format
```bash
make check        # check formatting with black + ruff (no changes)
make autoformat   # apply black + ruff fixes in place
make clean        # remove .pyc / __pycache__
```

### Smoke-test a framework module
Each framework file can be run standalone:
```bash
python starVLA/model/framework/QwenGR00T.py
python starVLA/model/framework/QwenOFT.py --config_yaml starvla_cotrain_oxe.yaml
python starVLA/dataloader/lerobot_datasets.py --config_yaml starvla_cotrain_oxe.yaml
```

### Training
```bash
# Single benchmark (LIBERO example):
bash examples/LIBERO/train_files/run_libero_train.sh

# General multi-GPU launch pattern:
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_libero_mapanything_llava3d.yaml \
  --framework.name MapAnythingLlava3DPI \
  --run_root_dir results/Checkpoints \
  --run_id my_run_id
```

CLI overrides work via dotlist syntax: `--framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct`

Freeze modules: `--trainer.freeze_modules "mapanythingllava3d_vlm_interface.model.vision_tower"`
(Use `print(model)` first to confirm module paths.)

### Evaluation (LIBERO two-terminal workflow)
```bash
# Terminal 1 (starVLA env): start inference server
bash examples/LIBERO/eval_files/run_policy_server.sh

# Terminal 2 (LIBERO env): run simulation
bash examples/LIBERO/eval_files/eval_libero.sh
```

### Deployment server
```bash
python deployment/model_server/server_policy.py \
    --ckpt_path results/Checkpoints/my_run/steps_50000_pytorch_model.pt \
    --port 10093 \
    --use_bf16

# Debug/test the server:
python deployment/model_server/debug_server_policy.py
```

## Architecture

### Top-level directory
```
starVLA/           # main Python package
  model/
    framework/     # end-to-end VLA orchestrators (one file per model)
    modules/
      vlm/         # VLM wrappers (QWen2.5, QWen3, Florence2, MapAnythingLlava3D)
      action_model/ # action heads (MLP, GR00T flow-matching, LayerwiseFM, FAST)
  dataloader/
    lerobot_datasets.py   # LeRobot dataset loader (returns raw dicts)
    vlm_datasets.py       # VLM multimodal dataset loader
    gr00t_lerobot/        # GR00T-style dataset utilities
  training/
    train_starvla.py        # VLA-only training loop
    train_starvla_cotrain.py # VLA + VLM co-training
    train_starvlm.py        # VLM-only training
    trainer_utils/          # TrainerUtils, build_param_lr_groups, config_tracker
  mapanything_llava3d/    # custom spatial VLM model code (2D+3D fusion)
    model/
      modeling_mapanything_llava3d_vlm.py  # main model
      modeling_mapanything.py              # geometric encoder wrapper
      processing_mapanything_llava3d.py    # processor / image token injection
  config/
    training/      # YAML config files per experiment
    deepseeds/     # DeepSpeed ZeRO configs
deployment/
  model_server/    # WebSocket policy server for evaluation
examples/          # benchmark-specific train + eval scripts
  LIBERO/
  SimplerEnv/
  Robocasa_tabletop/
  Behavior/
  Robotwin/
```

### Framework registry pattern
Every framework file registers itself via `@FRAMEWORK_REGISTRY.register("Name")`. The factory `build_framework(cfg)` in `starVLA/model/framework/__init__.py` auto-imports all framework modules and dispatches by `cfg.framework.name`. To add a new framework: create a file in `starVLA/model/framework/`, decorate the class, and it is discovered automatically.

### Supported frameworks
| Name | Action head | VLM |
|------|------------|-----|
| `QwenFast` | Discrete token (autoregressive) | Qwen2.5-VL |
| `QwenOFT` | MLP parallel regression | Qwen2.5-VL |
| `QwenPI` | Flow-matching (GR00T style) | Qwen2.5-VL |
| `QwenGR00T` | Flow-matching + dual-system | Qwen2.5-VL |
| `QwenAdapter` | Adapter-based | Qwen2.5-VL |
| `MapAnythingLlava3DPI` | Layerwise flow-matching (LayerwiseFM) | MapAnythingLlava3D (2D+3D) |

### MapAnythingLlava3D (this fork's custom VLM)
- **VLM interface**: `starVLA/model/modules/vlm/MapAnythingLlava3D.py`
- **Model backbone**: `starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`
- **Geometric encoder**: `starVLA/mapanything_llava3d/model/modeling_mapanything.py`
- **Processor**: `starVLA/mapanything_llava3d/model/processing_mapanything_llava3d.py`
- **Framework file**: `starVLA/model/framework/MapAnythingLlava3DPI.py`
- **Action head**: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`

The framework extracts **multi-layer hidden states** from the LLM and feeds them to the DiT-based cross-attention flow-matching head. `vl_layer_selection: first|last` controls which layer to use.

---

## Current Fusion Architecture (MapAnythingLlava3DPI) — Active Research Area

This is the primary area of active development. Understanding the full data flow is essential.

### What a "batch" is
A batch is a group of B samples processed together. Each sample is one robot observation: a set of images + one language instruction + state + action labels. All tensor shapes have `B` as their first dimension. For example `[B=4, 256, 4096]` means 4 separate observations, each with 256 patch tokens of dimension 4096.

### How images become patch tokens (SigLIP)
SigLIP cuts each 224×224 image into 14×14 pixel patches → 16×16 = **256 patches per image**. Each patch gets one feature vector of dim 1152. For V views (e.g. 2 cameras), views are stacked to give `V*256` tokens per sample:
```
pixel_values [B, V, 3, 224, 224]
  → SigLIP vision_tower → [B*V, 256, 1152]
  → vision_projector (Linear 1152→H) → [B*V, 256, H]
  → reshape → [B, V*256, H]     e.g. [B, 512, 4096] for 2 views, H=4096
```
Each token at position `(row, col)` in the 16×16 grid carries **spatially specific** semantic information about that exact patch of the image.

### How geometric features are produced (MapAnything)
MapAnything receives the same images but processes them differently — it extracts 3D-aware features using a DINOv2 encoder followed by a multi-view cross-attention transformer (`info_sharing`). Its output is also a sequence of spatially structured feature vectors:
```
pixel_values → SigLIP-to-DINOv2 de-norm: (x * 0.5 + 0.5).clamp(0,1)
  → MapAnythingWrapper._encode_n_views → per-view DINOv2 features
  → map_anything_model.info_sharing (multi-view transformer) → [B, G, geom_dim]
```
`G` is the geometric sequence length (depends on MapAnything's internal resolution; likely different from V*256). Each of the G tokens encodes **spatially specific** 3D information — depth, surface normals, multi-view correspondences — for one spatial location.

### The current fusion (the known bottleneck)

All fusion happens in `fusion_module()` at `modeling_mapanything_llava3d_vlm.py:253`. This is currently a **placeholder** (marked with a `TODO` comment in the code):

```
Step 1 — Project geometry to LLM dim:
  [B, G, geom_dim]  →  geometric_projector (Linear)  →  [B, G, H]

Step 2 — Global average pool across all G positions:
  [B, G, H]  →  .mean(dim=1)  →  [B, 1, H]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ALL spatial 3D structure is destroyed here.
  G different per-location vectors become ONE averaged scalar vector per sample.

Step 3 — Broadcast that one vector to every vision patch position:
  [B, 1, H]  →  .expand(B, V*256, H)  →  [B, V*256, H]
  Every patch (the cup patch, the table patch, the wall patch)
  receives THE SAME averaged geometric context.

Step 4 — Concatenate with vision features and project:
  cat([vision [B,V*256,H], geom_broadcast [B,V*256,H]], dim=-1) → [B, V*256, 2H]
  →  fusion_projector (Linear 2H→H)  →  fused_tokens [B, V*256, H]
```

The result `[B, V*256, H]` are the fused visual tokens that get injected into the LLM at `<image>` token positions.

### Where the three modalities meet

| Stage | Location | What happens |
|---|---|---|
| Early fusion | `get_image_features()` → `fusion_module()` | Geometry (globally pooled, 1 vector) is broadcast and concatenated with each vision patch. Language is NOT present here. |
| LLM input | `forward()` line 332 | Fused tokens injected at `<image>` positions. LLM sequence = `[lang_tokens, fused_visual_tokens×V*256, lang_tokens]`. |
| LLM self-attention | Inside LLaVA3D LLM | Language and fused visual tokens cross-modify each other via standard attention. Language can influence vision but cannot selectively address geometry (it was already collapsed). |
| Action head | `LayerwiseFM_ActionHeader` | K DiT blocks each cross-attend to one of the last K LLM hidden-state layers (`vl_embs_list`). The full LLM sequence (lang + fused vision+geom) is the cross-attention context. |

### Known structural problems with current fusion
1. **Spatial 3D information is destroyed** — Mean pooling collapses G spatial geometric tokens to 1 average vector. The per-position depth/normal information that MapAnything computed is thrown away.
2. **Every visual patch sees identical geometry** — After broadcasting, patch (2,2) looking at a cup and patch (8,8) looking at the table receive the exact same geometric context.
3. **Language cannot guide geometric selection** — Fusion happens before tokenization. Whether the task says "pick up the red cup" or "open the drawer", geometry is pooled identically.
4. **G ≠ V*256** — MapAnything and SigLIP produce sequences of different lengths, so token-wise spatial fusion requires explicit alignment.

### Ablation config map
The configs in `starVLA/config/training/` encode the design axes being explored:
- `starvla_train_libero_mapanything_llava3d_ab_a_pure_cross.yaml` — **no geometry** (`use_geometric_branch: false`), pure VL layerwise cross-attention in DiT (`use_concat_cross_context: false`). Baseline.
- `starvla_train_libero_mapanything_llava3d_ab_b_concat_cross.yaml` — **no geometry**, concat context in DiT cross-attention (`use_concat_cross_context: true`).
- `starvla_train_libero_mapanything_llava3d_ab_b_concat_cross_geometric.yaml` — **with geometry** (`use_geometric_branch: true`) + concat context.
- `starvla_train_libero_mapanything_llava3d_ab_b_concat_cross_geometric_alg1.yaml` — same as above + auxiliary world-model losses (`loss_w_dyn`, `loss_w_geo`, `loss_w_reg`) and a phased training schedule (`sagr_phase_a_steps`, `sagr_phase_b_ramp_steps`). Not yet fully implemented in action head.

Note: the `ab_a` and `ab_b` configs both have `use_geometric_branch: false` — they are ablating the DiT cross-attention strategy, not the geometric branch itself. Only the `_geometric` suffix configs actually enable the geometric encoder.

### Configuration system
All configs are OmegaConf YAML. The single entry point is `--config_yaml`. CLI dotlist args override any YAML key. Key config sections:
- `framework.name` — selects which framework to build
- `framework.mapanything_llava3d.base_vlm` — path to pretrained VLM
- `trainer.freeze_modules` — comma-separated regex/name list for frozen modules
- `trainer.learning_rate` — nested dict mapping module name prefix → lr (matched by substring)
- `trainer.pretrained_checkpoint` + `trainer.reload_modules` — resume/partial load

### Dataloader contract
Dataloaders return **raw, model-agnostic dicts** only — no tokenization or image encoding. The framework's `forward()` and `predict_action()` handle all preprocessing. A sample includes: `image` (list[PIL.Image] or np.ndarray), `lang` (str), `action` (np.ndarray), `state` (optional).

### Checkpoint format
Checkpoints are saved as `steps_{N}_pytorch_model.pt` (raw `state_dict`). A companion `config.yaml` and `norm_stats.json` are saved alongside each checkpoint. Loading uses `baseframework.from_pretrained(path)`.

## Pretrained Model Paths (this environment)
- MapAnythingLlava3D base model: `/2025233147/zzq/SpatialVLA_llava3d/model_zoo/mapanythingllava3d_base_v3`
- LIBERO dataset: `/2025233147/zzq/SpatialVLA_llava3d/playground/Datasets/LEROBOT_LIBERO_DATA`
- Qwen models: `./playground/Pretrained_models/`
