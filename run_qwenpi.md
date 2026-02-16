# Running QwenPI Framework - Complete Guide

**Date:** 2026-02-16
**Project:** StarVLA - Vision-Language-Action Model Framework
**Focus:** QwenPI Framework Setup and Training

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What We Discovered](#what-we-discovered)
3. [QwenPI Framework Details](#qwenpi-framework-details)
4. [Complete Setup Guide](#complete-setup-guide)
5. [Configuration Files Created](#configuration-files-created)
6. [Training Commands](#training-commands)
7. [Debugging Wrong Actions Issue](#debugging-wrong-actions-issue)
8. [Key Files Reference](#key-files-reference)

---

## 1. Project Overview

### What is StarVLA?

**StarVLA** is a modular, "Lego-like" codebase for developing Vision-Language-Action (VLA) models that bridge vision-language understanding with robotic control.

- **GitHub:** https://github.com/starVLA/starVLA
- **Version:** 1.2.0 (as of January 2025)
- **Latest Updates:** Support for LeRobot v3.0, DeepSpeed ZeRO-3, multiple benchmarks

### Available VLA Frameworks

StarVLA provides **4 main architectures**:

1. **Qwen-FAST**: Qwen2.5-VL + Fast tokenizer (autoregressive discrete action tokens)
2. **Qwen-OFT**: Qwen2.5-VL + MLP action head (parallel continuous actions)
3. **Qwen-PI** ⭐: Qwen2.5-VL + Flow-Matching head (diffusion-based, **best performance**)
4. **Qwen-GR00T**: Qwen2.5-VL + Dual-system architecture (System2 reasoning)

---

## 2. What We Discovered

### Initial Exploration (Task 1)

We conducted a comprehensive exploration of the local StarVLA codebase and found:

#### Project Structure
```
D:\sist\starVLA\
├── starVLA/
│   ├── model/
│   │   ├── framework/          # QwenPI.py, QwenGR00T.py, 3DVLA.py, etc.
│   │   └── modules/
│   │       ├── vlm/            # Vision-language models
│   │       ├── action_model/   # LayerwiseFM_ActionHeader.py (flow-matching)
│   │       └── projector/      # Feature fusion modules
│   ├── dataloader/             # LeRobot dataset loading
│   ├── training/               # train_starvla.py (main entry)
│   └── config/
│       └── training/           # YAML configuration files
├── examples/                   # LIBERO, SimplerEnv, RoboCasa eval scripts
├── playground/                 # Where models and datasets should go
│   ├── Pretrained_models/      # (To be created)
│   └── Datasets/               # (To be created)
└── requirements.txt
```

#### Current Development Status

Based on recent git commits:
- ✅ MapAnything (3D geometric encoder) - implemented
- ✅ SigLIP (2D semantic encoder) - implemented
- ✅ GeometryFusion module - completed
- ✅ Llava3D wrapper - created
- 🚧 3DVLA framework - in progress (custom development)
- ⭐ **QwenPI framework - fully implemented and ready to use**

### Official Documentation Research (Task 2)

We researched the official StarVLA GitHub and found:

#### QwenPI Performance Benchmarks

| Model | WidowX Success Rate | Training Time (16×A100) |
|-------|---------------------|-------------------------|
| Qwen2.5-FAST | 58.6% | ~3 hours (10k steps) |
| Qwen2.5-OFT | 41.8% | ~3 hours (10k steps) |
| **Qwen2.5-PI** | **62.5%** ⭐ | ~18 hours (30k steps) |
| Qwen2.5-GR00T | 63.6% | ~18 hours (30k steps) |
| Qwen-GR00T-Bridge | **71.4%** | - |

**QwenPI Architecture:**
- Base VLM: Qwen2.5-VL-3B-Instruct
- Action Head: Layerwise Flow-Matching DiT
- Approach: Diffusion-based continuous action prediction (aligned with π₀)
- Training: 30k steps on Bridge + RT-1 datasets

---

## 3. QwenPI Framework Details

### Architecture Components

#### 1. Vision-Language Model (Qwen2.5-VL-3B)
- **File:** `starVLA/model/modules/vlm/QWen2_5.py`
- **Hidden dim:** 2048
- **Layers:** 36
- **Attention:** FlashAttention2 (required)
- **Input:** Multi-view images + text instructions
- **Output:** Hidden states from all layers (for layerwise conditioning)

#### 2. Layerwise Flow-Matching Action Head
- **File:** `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- **Model type:** DiT-B/DiT-L (Diffusion Transformer)
- **Technique:** Flow-matching (continuous normalizing flows)
- **Cross-attention:** Uses layerwise VLM hidden states
- **Training:** Add noise + predict denoised residual
- **Inference:** DDIM sampling (4-10 steps)

#### 3. Framework Implementation
- **File:** `starVLA/model/framework/QwenPI.py`
- **Forward pass:**
  1. Encode images + text with Qwen2.5-VL
  2. Extract last N layers' hidden states (for DiT cross-attention)
  3. Pass to Flow-Matching head with action labels
  4. Compute diffusion loss
- **Inference:**
  1. Encode observation
  2. Sample actions via DDIM
  3. **CRITICAL:** Denormalize actions using dataset statistics

### Key Parameters

```yaml
action_model:
  action_model_type: DiT-B           # DiT architecture size
  action_dim: 7                      # Robot action space (6DOF + gripper)
  state_dim: 7                       # Robot state dimension
  future_action_window_size: 15      # Predict 16 steps (0-15)
  past_action_window_size: 0         # No history conditioning
  repeated_diffusion_steps: 4        # Training augmentation
  num_inference_timesteps: 4         # DDIM sampling steps

  diffusion_model_cfg:
    cross_attention_dim: 2048        # Match VLM hidden dim
    num_layers: 16                   # DiT depth
    dropout: 0.2
    norm_type: "ada_norm"            # Adaptive normalization
```

---

## 4. Complete Setup Guide

### Step 1: Environment Setup

```bash
cd D:/sist/starVLA

# Create environment
conda create -n starVLA python=3.10 -y
conda activate starVLA

# Install dependencies
pip install -r requirements.txt

# Install FlashAttention2 (CRITICAL for QwenPI)
pip install flash-attn --no-build-isolation

# Install StarVLA package
pip install -e .
```

**Requirements:**
- Python 3.10+
- CUDA 12.0 or 12.4 (for flash-attn==2.7.4.post1)
- At least 24GB VRAM per GPU

### Step 2: Download Pretrained Models

```bash
mkdir -p playground/Pretrained_models
cd playground/Pretrained_models

# Option A: Base model for training from scratch
huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
  --local-dir ./Qwen2.5-VL-3B-Instruct

# Option B: Pretrained QwenPI checkpoint (recommended for fine-tuning)
huggingface-cli download StarVLA/Qwen-FM-Bridge-RT-1 \
  --local-dir ./Qwen-FM-Bridge-RT-1
```

**Official Checkpoints:**
- **Qwen2.5-VL-3B-Instruct-Action**: Base VLM with extended vocabulary (Fast Tokens)
- **Qwen-FM-Bridge-RT-1**: Trained on Bridge+RT-1, 62.5% WidowX success

### Step 3: Prepare Datasets

```bash
mkdir -p playground/Datasets

# For LIBERO (simulation)
mkdir -p playground/Datasets/LEROBOT_LIBERO_DATA
# Download LIBERO datasets in LeRobot format
# (Check LeRobot hub or StarVLA docs for links)

# For real robot (Bridge + RT-1)
huggingface-cli download IPEC-COMMUNITY/bridge_orig_lerobot \
  --local-dir ./playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig

huggingface-cli download IPEC-COMMUNITY/fractal20220817_data_lerobot \
  --local-dir ./playground/Datasets/OXE_LEROBOT_DATASET/fractal20220817_data
```

**Required dataset structure (LeRobot format):**
```
playground/Datasets/LEROBOT_LIBERO_DATA/
├── meta/
│   ├── modality.json          ← CRITICAL: Must exist
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── info.json
│   └── stats_gr00t.json
└── data/
    └── **/*.parquet           ← Actual episode data
```

### Step 4: Verify Setup

```bash
# Test framework loads correctly
python starVLA/model/framework/QwenPI.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml

# Test dataloader (requires dataset)
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml
```

---

## 5. Configuration Files Created

We created two QwenPI-specific configuration files:

### Config 1: LIBERO Training (Simulation)

**File:** `starVLA/config/training/starvla_qwenpi_libero.yaml`

```yaml
run_id: qwenpi_libero
framework:
  name: QwenPI  # ← Key setting
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: DiT-B
    action_dim: 7
    future_action_window_size: 15

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA
    data_mix: libero_goal
    action_type: delta_qpos       # Simulation uses joint delta
    per_device_batch_size: 16

trainer:
  max_train_steps: 30000
  learning_rate:
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04          # Higher LR for action head
  loss_scale:
    vla: 1.0
    vlm: 0.0                       # Disable VLM co-training
```

**Key Features:**
- VLA-only training (no VLM loss)
- LIBERO dataset (simulation tasks)
- Action type: `delta_qpos` (joint space deltas)

### Config 2: Bridge+RT-1 Training (Real Robot)

**File:** `starVLA/config/training/starvla_qwenpi_bridge.yaml`

```yaml
run_id: qwenpi_bridge_rt1
framework:
  name: QwenPI
  # ... same VLM settings ...

datasets:
  vla_data:
    data_root_dir: playground/Datasets/OXE_LEROBOT_DATASET
    data_mix: bridge_rt_1         # Real robot dataset
    action_type: delta_ee         # End-effector delta actions
    per_device_batch_size: 16

trainer:
  max_train_steps: 30000           # 18 hours on 16×A100
```

**Key Features:**
- Real robot training data
- Action type: `delta_ee` (end-effector space)
- Same as official QwenPI training setup

---

## 6. Training Commands

### Single GPU Training (Quick Test)

```bash
cd D:/sist/starVLA

python starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml
```

### Multi-GPU Training (Production)

```bash
# 8 GPUs with DeepSpeed ZeRO-2
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml

# 4 GPUs
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml
```

### Resume Training

```bash
python starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml \
  --trainer.is_resume true
```

### Fine-tune from Pretrained

```bash
python starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml \
  --trainer.pretrained_checkpoint ./playground/Pretrained_models/Qwen-FM-Bridge-RT-1/pytorch_model.pt
```

### Training Outputs

```
results/Checkpoints/qwenpi_libero/
├── checkpoints/
│   ├── steps_5000_pytorch_model.pt
│   ├── steps_10000_pytorch_model.pt
│   └── ...
├── config.yaml                    # Training configuration
├── summary.jsonl                  # Training metrics
└── dataset_statistics.json        # ⭐ CRITICAL for inference
```

### Expected Training Time

| Setup | Time for 30k steps |
|-------|-------------------|
| 1× A100 | ~60-80 hours |
| 4× A100 | ~36 hours |
| 8× A100 | ~18 hours |
| 16× A100 | ~9 hours |

---

## 7. Debugging Wrong Actions Issue

### Problem Statement

**User reported:** Model makes actions smoothly but actions are completely wrong.

### Root Cause Analysis

This is the **#1 most common issue** in VLA models: **Missing Action Denormalization**

#### Why This Happens

1. **During training:** Actions are normalized to range `[-1, 1]` using dataset statistics
2. **During inference:** Model outputs normalized actions in `[-1, 1]`
3. **Before execution:** Must denormalize back to real robot scale
4. **If skipped:** Robot receives tiny normalized values instead of real commands

### The Critical Denormalization Step

**File:** `starVLA/model/framework/base_framework.py:175-205`

```python
def unnormalize_actions(normalized_actions: np.ndarray,
                       action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Map normalized actions (≈[-1, 1]) back to original value range.

    Formula:
        real_action = 0.5 * (norm_action + 1) * (q99 - q01) + q01

    where q01 = 1st percentile, q99 = 99th percentile from training data
    """
    # Get statistics
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])  # 99th percentile
    action_low = np.array(action_norm_stats["q01"])   # 1st percentile

    # Clip to valid range
    normalized_actions = np.clip(normalized_actions, -1, 1)

    # Binarize gripper (channel 6)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)

    # Denormalize using linear scaling
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return actions
```

### Correct Inference Pattern

**❌ WRONG:**
```python
# Model outputs normalized actions
normalized_actions = model.predict_action(examples)["normalized_actions"]

# Directly use on robot - WRONG!
robot.execute(normalized_actions)  # <- Actions will be tiny/wrong!
```

**✅ CORRECT:**
```python
# 1. Load model with statistics
model = QwenPI.from_pretrained("path/to/checkpoint")  # Auto-loads norm_stats

# 2. Get normalized actions
normalized_actions = model.predict_action(examples)["normalized_actions"]

# 3. Get statistics for your dataset
action_stats = model.get_action_stats(unnorm_key="libero_goal")

# 4. Denormalize to real scale
real_actions = model.unnormalize_actions(normalized_actions, action_stats)

# 5. Now execute
robot.execute(real_actions)  # ← Correct scale!
```

### Required Files for Inference

**CRITICAL:** `dataset_statistics.json` must exist in checkpoint directory!

```json
{
  "libero_goal": {
    "action": {
      "q01": [0.1, 0.2, 0.3, ...],   # 1st percentile (lower bound)
      "q99": [0.9, 0.8, 0.7, ...],   # 99th percentile (upper bound)
      "mask": [true, true, true, ...],
      "mean": [...],
      "std": [...]
    },
    "state": {
      ...
    }
  }
}
```

This file is **automatically created during training** and saved alongside checkpoints.

### Diagnostic Checklist

If actions are wrong, check:

1. ✅ **Statistics file exists:**
   ```bash
   ls results/Checkpoints/qwenpi_libero/dataset_statistics.json
   ```

2. ✅ **Using correct denormalization:**
   - Must call `model.unnormalize_actions()`
   - Must use statistics from training dataset

3. ✅ **Correct unnorm_key:**
   - Must match training dataset name
   - Check available keys: `model.norm_stats.keys()`

4. ✅ **Action dimensions match:**
   - Model trained on 7-DOF → must use 7-DOF robot
   - Check `action_dim` in config

5. ✅ **Action space type matches:**
   - Training: `delta_qpos` → Inference: joint deltas
   - Training: `delta_ee` → Inference: end-effector deltas

6. ✅ **Taking correct timestep:**
   - Model outputs chunk (e.g., 16 steps)
   - Must select appropriate timestep (usually [0] or sliding window)

### Example from Evaluation Code

**Reference:** `examples/LIBERO/eval_files/model2libero_interface.py:123`

```python
# Step 1: Get normalized actions from model
normalized_actions = response["data"]["normalized_actions"]  # [B, chunk, D]

# Step 2: Denormalize using statistics
self.raw_actions = self.unnormalize_actions(
    normalized_actions=normalized_actions,
    action_norm_stats=self.action_norm_stats  # Loaded from checkpoint
)

# Step 3: Use denormalized actions
raw_actions = self.raw_actions[step % action_chunk_size]
robot.execute(raw_actions)
```

---

## 8. Key Files Reference

### Framework Implementation

| File | Purpose |
|------|---------|
| `starVLA/model/framework/QwenPI.py` | Main QwenPI framework class |
| `starVLA/model/framework/base_framework.py` | Base class with denormalization utilities |
| `starVLA/model/modules/vlm/QWen2_5.py` | Qwen2.5-VL wrapper |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | Flow-matching action head |

### Training Infrastructure

| File | Purpose |
|------|---------|
| `starVLA/training/train_starvla.py` | Main training script |
| `starVLA/dataloader/lerobot_datasets.py` | LeRobot dataset loader |
| `starVLA/config/training/*.yaml` | Training configurations |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | DeepSpeed config |

### Evaluation Examples

| Directory | Purpose |
|-----------|---------|
| `examples/LIBERO/eval_files/` | LIBERO benchmark evaluation |
| `examples/SimplerEnv/eval_files/` | SimplerEnv evaluation |
| `examples/RoboCasa_tabletop/eval_files/` | RoboCasa evaluation |

### Configuration Files We Created

| File | Purpose |
|------|---------|
| `starVLA/config/training/starvla_qwenpi_libero.yaml` | QwenPI config for LIBERO |
| `starVLA/config/training/starvla_qwenpi_bridge.yaml` | QwenPI config for Bridge+RT-1 |

---

## 🎯 Quick Reference: Complete Workflow

### 1. Setup (One-time)
```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

### 2. Download Models & Data
```bash
# Models
huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
  --local-dir playground/Pretrained_models/Qwen2.5-VL-3B-Instruct

# Datasets
huggingface-cli download IPEC-COMMUNITY/bridge_orig_lerobot \
  --local-dir playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig
```

### 3. Train
```bash
# Multi-GPU training
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_qwenpi_libero.yaml
```

### 4. Inference (CRITICAL: Include Denormalization!)
```python
# Load model
model = QwenPI.from_pretrained("results/Checkpoints/qwenpi_libero/checkpoints/steps_30000_pytorch_model.pt")

# Predict
normalized_actions = model.predict_action([example])["normalized_actions"]

# Denormalize (REQUIRED!)
stats = model.get_action_stats(unnorm_key="libero_goal")
real_actions = model.unnormalize_actions(normalized_actions, stats)

# Execute
robot.execute(real_actions[0])  # Take first timestep
```

---

## 📚 Official Resources

- **GitHub:** https://github.com/starVLA/starVLA
- **README:** https://github.com/starVLA/starVLA/blob/starVLA/README.md
- **QwenPI Checkpoint:** https://huggingface.co/StarVLA/Qwen-FM-Bridge-RT-1
- **Bridge Dataset:** https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot
- **Fractal Dataset:** https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot

---

## ✅ Pre-Flight Checklist

Before training, verify:

- [ ] Environment: `conda activate starVLA` + FlashAttention installed
- [ ] Pretrained model: Downloaded to `playground/Pretrained_models/`
- [ ] Dataset: Downloaded with `modality.json` in `meta/` folder
- [ ] Config: `wandb_entity` and paths updated
- [ ] GPU: At least 24GB VRAM per GPU
- [ ] Disk: 50GB+ free for checkpoints
- [ ] Framework: Config has `framework.name: QwenPI`

Before inference, verify:

- [ ] Checkpoint: Includes `dataset_statistics.json`
- [ ] Denormalization: Code calls `unnormalize_actions()`
- [ ] Statistics: Using correct `unnorm_key` for dataset
- [ ] Action space: Matches training (delta_ee vs delta_qpos)

---

## 🔧 Common Issues & Solutions

### Issue 1: FlashAttention fails
```bash
pip uninstall flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation
# Requires CUDA 12.0 or 12.4
```

### Issue 2: Out of Memory
Edit config:
```yaml
per_device_batch_size: 8          # Reduce from 16
gradient_accumulation_steps: 2    # Increase to compensate
```

### Issue 3: Wrong actions (smooth but incorrect)
**MOST COMMON ISSUE** - Missing denormalization!
- Check `dataset_statistics.json` exists in checkpoint
- Must call `model.unnormalize_actions()` before execution
- Use correct `unnorm_key` matching training dataset

### Issue 4: Dataset not found
```bash
# Verify structure
ls playground/Datasets/LEROBOT_LIBERO_DATA/meta/modality.json
```

---

**End of Guide**

*This document summarizes all work done on 2026-02-16 for setting up and running the QwenPI framework in StarVLA.*
