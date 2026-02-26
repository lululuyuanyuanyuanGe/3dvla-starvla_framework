# Training Plan: 3D Geometry Injection Ablation on LIBERO

## Goal

Compare 3 geometry compression strategies for injecting MapAnything 3D features directly into the DiT action head, bypassing the LLM bottleneck.

| Strategy | Key Idea | Compressor Params | GPU |
|----------|----------|-------------------|-----|
| **pool** (Approach A) | Linear projection + group pooling → same K=16 tokens at every DiT layer | 3.15M | GPU 0 |
| **per_layer** (Approach B) | N=16 separate Linear projections, one per DiT layer → different tokens per layer | 50.36M | GPU 1 |
| **qformer** (Approach C) | K=16 learnable queries cross-attend over G geometry tokens → input-adaptive | 19.97M | GPU 2 |

**Hypothesis**: Spatially-specific geometry tokens in the DiT cross-attention will improve action accuracy compared to the current approach (global average pooling through the LLM).

---

## Architecture: QwenMapAnythingPI

```
                     Images [B, V, 3, 224, 224]
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
      Qwen2.5-VL (semantic)        MapAnything (geometric)
      SigLIP + LLM backbone        DINOv2 + multi-view transformer
              │                            │
              ▼                            ▼
    vl_embs_list [N layers]         raw_geom [B, G, geom_dim=1536]
    each [B, seq_len, H=2048]             │
              │                            ▼
              │                    Geometry Compressor
              │                    (pool | per_layer | qformer)
              │                            │
              │                            ▼
              │                    geom_tokens [B, K=16, H=2048]
              │                            │
              └──────────┬─────────────────┘
                         ▼
               DiT Action Head (16 layers)
               Each layer: cross_context = cat(vl_embs[i], geom_tokens)
                         │
                         ▼
                 Action prediction [B, T=8, action_dim=7]
```

---

## Data Pipeline

```
Config YAML
  data_root_dir: /2025233147/zzq/SpatialVLA_llava3d/playground/Datasets/LEROBOT_LIBERO_DATA
  data_mix: libero_all
      │
      ▼
mixtures.py → DATASET_NAMED_MIXTURES["libero_all"]
  ├── libero_object_no_noops_1.0.0_lerobot   (454 episodes, 10 tasks)
  ├── libero_goal_no_noops_1.0.0_lerobot     (428 episodes, 10 tasks)
  ├── libero_spatial_no_noops_1.0.0_lerobot  (432 episodes, 10 tasks)
  └── libero_10_no_noops_1.0.0_lerobot       (379 episodes, 10 tasks)
      │
      ▼
lerobot_datasets.py → LeRobotMixtureDataset
  Returns raw dicts: {"image": [PIL], "lang": str, "action": ndarray, "state": ndarray}
      │
      ▼
QwenMapAnythingPI.forward()
  - Images → Qwen2.5-VL tokenizer + MapAnything encoder
  - Language → Qwen2.5-VL tokenizer
  - Action/State → DiT action head (flow matching loss)
```

Total: ~1693 episodes, ~273k frames, 40 tasks across 4 LIBERO suites.

---

## Training Configuration

### Shared settings (all 3 runs)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Framework | QwenMapAnythingPI | Qwen2.5-VL + standalone MapAnything |
| base_vlm | `/2025233147/.../qwen2.5vl` | Qwen2.5-VL-3B local |
| mapanything_model_path | `/2025233147/.../mapanything` | Pre-trained MapAnything |
| data_mix | `libero_all` | All 4 LIBERO suites |
| per_device_batch_size | 4 | Single GPU, matches existing LIBERO scripts |
| gradient_accumulation_steps | 4 | Effective batch = 16 |
| max_train_steps | 80,000 | Matches existing LIBERO training |
| save_interval | 10,000 | 8 checkpoints total |
| seed | 42 | Fair comparison |
| DiT layers | 16 | Standard DiT-B |
| geom_dit_tokens (K) | 16 | Compressed geometry sequence length |
| action_dim | 7 | 6-DoF + gripper |
| action_horizon | 8 | Predict 8 future steps (matches LIBERO action chunks) |
| future_action_window_size | 7 | action_horizon - 1 |

### Freezing strategy

| Module | Params | Frozen? | Rationale |
|--------|--------|---------|-----------|
| qwen_vl_interface | 3,754.62M | No (LR 1e-5) | Fine-tuned at low LR so VLM adapts to robot domain. |
| mapanything_encoder | 1,228.49M | **Yes** | Pre-trained 3D encoder. Frozen so all 3 runs receive identical geometric features → clean ablation. |
| action_model (DiT) | 977.75M | No (LR 1e-4) | DiT action head, must train from scratch. |
| geom_compressor | 3.15 / 50.36 / 19.97M | No (LR 1e-4) | The variable under test (pool / per_layer / qformer). |

Total model size and trainable params per strategy:

| Strategy | Total Params | Trainable (after freeze) |
|----------|-------------|--------------------------|
| pool | 5,964M | ~4,736M |
| qformer | 5,981M | ~4,752M |
| per_layer | 6,011M | ~4,783M |

Config: `freeze_modules: 'mapanything_encoder'`

### MapAnything Feature Caching

Since MapAnything is frozen, the same input image always produces identical features. An in-memory cache eliminates redundant 1.2B-parameter forward passes after the first epoch.

**How it works:**
1. Each image is MD5-hashed to produce a cache key
2. On cache hit (all samples cached): skip MapAnything entirely, return stacked cached tensors
3. On cache miss: run MapAnything, store each sample's features on CPU in bf16

**Memory budget:**
- Per frame: 196 tokens × 1536 dim × 2 bytes (bf16) = ~600KB
- Full LIBERO dataset (~273k frames): ~160GB per process
- 3 processes: ~480GB total (fits in 2TB system RAM)

**Performance:**
- Epoch 1 (cold cache): same speed as before (~5s/step)
- Epoch 2+ (warm cache): MapAnything skipped entirely (~30% faster, ~3.5s/step)

**Cache stats** are logged periodically:
```
[GeomCache] 50000 entries, hit rate: 45.2% (50000/110600)
```

Implementation: `QwenMapAnythingPI._run_mapanything()` in `starVLA/model/framework/QwenMapAnythingPI.py`

### Learning rates

| Module | LR | Why |
|--------|-----|-----|
| base (default) | 1e-5 | Conservative for pre-trained modules |
| qwen_vl_interface | 1e-5 | Fine-tune VLM slowly |
| mapanything_encoder | 1e-5 | N/A (frozen, LR ignored) |
| action_model | 1e-4 | Action head trains faster (10x) |
| geom_compressor | 1e-4 | New module, needs faster learning |

### Per-strategy differences

| | pool | per_layer | qformer |
|---|---|---|---|
| `geom_compression_type` | pool | per_layer | qformer |
| `geom_qformer_heads` | N/A | N/A | 8 |
| `run_id` | `qwen_mapanything_pool` | `qwen_mapanything_per_layer` | `qwen_mapanything_qformer` |
| GPU | 0 | 1 | 2 |

---

## Config Files

```
starVLA/config/training/
  starvla_train_libero_qwen_mapanything_pool.yaml       ← GPU 0
  starvla_train_libero_qwen_mapanything_per_layer.yaml   ← GPU 1
  starvla_train_libero_qwen_mapanything_qformer.yaml     ← GPU 2

starVLA/config/deepseeds/
  deepspeed_zero2_1gpu.yaml   ← single-GPU accelerate config
```

---

## Launch

### Single command (all 3 in parallel)
```bash
# With WandB visualization:
WANDB_MODE=online bash examples/LIBERO/train_files/run_qwen_mapanything_3gpu.sh

# Without WandB:
bash examples/LIBERO/train_files/run_qwen_mapanything_3gpu.sh
```

### Per-run launch pattern
```bash
CUDA_VISIBLE_DEVICES=X accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_1gpu.yaml \
  --num_processes 1 \
  --gradient_accumulation_steps 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_libero_qwen_mapanything_<strategy>.yaml \
  ...
```

### Monitoring
```bash
# GPU usage
watch -n 5 nvidia-smi

# Training logs (run_id includes seed and timestamp)
tail -f results/Checkpoints/qwen_mapanything_pool_s42_*/train.raw.log
tail -f results/Checkpoints/qwen_mapanything_per_layer_s42_*/train.raw.log
tail -f results/Checkpoints/qwen_mapanything_qformer_s42_*/train.raw.log

# WandB dashboard (all 3 runs in one project):
# https://wandb.ai/gelululuyuan-ge/qwen_mapanything_geom_ablation

# RAM usage (monitor geom cache growth):
watch -n 30 free -h
```

### Debugging

```bash
# Kill stuck training processes
pkill -f "train_starvla.py"

# Clean up failed runs
rm -rf results/Checkpoints/qwen_mapanything_*

# Check if processes are still running
ps aux | grep train_starvla

# Check GPU status
nvidia-smi

# Restart
WANDB_MODE=online bash examples/LIBERO/train_files/run_qwen_mapanything_3gpu.sh
```

---

## Checkpoints

Saved at: `results/Checkpoints/<run_id>/steps_XXXXX_pytorch_model.pt`

Each checkpoint includes:
- `steps_XXXXX_pytorch_model.pt` — full model state_dict
- `config.yaml` — training config snapshot
- `norm_stats.json` — normalization statistics

Checkpoints at steps: 10k, 20k, 30k, 40k, 50k, 60k, 70k, 80k

---

## Evaluation (after training)

Use the LIBERO two-terminal workflow:

```bash
# Terminal 1: start policy server with a checkpoint
python deployment/model_server/server_policy.py \
    --ckpt_path results/Checkpoints/qwen_mapanything_pool/steps_80000_pytorch_model.pt \
    --port 10093 --use_bf16

# Terminal 2: run LIBERO simulation
bash examples/LIBERO/eval_files/eval_libero.sh
```

Compare success rates across pool vs per_layer vs qformer on each LIBERO suite.

---

## Expected Outcomes

| Metric | Where to check |
|--------|---------------|
| Training loss curve | WandB dashboard or `results/Checkpoints/<run_id>/metrics.jsonl` |
| Gradient flow to geom_compressor | Training logs (logged by trainer) |
| GPU memory usage | `nvidia-smi` (~125GB per run on H20-144GB) |
| GeomCache hit rate | Training logs: `[GeomCache] N entries, hit rate: X%` |
| RAM usage (cache) | `free -h` (~160GB per process after first epoch) |
| LIBERO success rate | Evaluation after training |

## Bugs Fixed During Setup

| Bug | Symptom | Fix |
|-----|---------|-----|
| VLM routing | `NotImplementedError: VLM model qwen2.5vl` | Case-insensitive match in `vlm/__init__.py` |
| Action horizon | `predicted.shape=(4,16,7) vs ground_truth=(4,8,7)` | `action_horizon: 16` → `8` (match LIBERO) |
| Port conflict | `EADDRINUSE` on port 29500 | Separate `MASTER_PORT` per run (29500/29501/29502) |
| Config namespace | `Missing key num_vl_layers` | Mirror VL dims to `qwenvl` namespace |
| Dtype mismatch | `Float vs BFloat16` in geom compressor | Cast `raw_geom` to compressor weight dtype |
| Eval hang | Training stuck at eval steps | `eval_interval: 100` → `5000` |
| data_mix wrong | Config pointed to OXE instead of LIBERO | `bridge_rt_1` → `libero_all` |
| wandb_entity | 404 error on upload | `bbbforbazinga` → `gelululuyuan-ge` |
