# Research Plan: Direct 3D Geometry Injection into DiT Action Head

## Hypothesis

The robot understands tasks semantically (correct object selection, correct intent) but fails on precise 3D control (grasping accuracy, spatial trajectory). This is because 3D geometric information from MapAnything is destroyed by mean-pooling before reaching the action head. Injecting compressed geometric features **directly** into the DiT's cross-attention context — bypassing the LLM — will improve control accuracy.

## Architecture

```
Semantic path (existing, works well):
  Images + Instruction --> Qwen2.5-VL --> multi-layer hidden states --> DiT cross-attn

Geometry path (NEW):
  Images --> MapAnything (standalone DINOv2 + multi-view transformer)
         --> G spatial tokens [B, G, 1024]
         --> Compress to K tokens [B, K, 2048]
         --> Concatenate into DiT cross-attn context at every layer

Combined DiT cross-attention context per layer:
  context = cat(vl_hidden_states[layer_i], geom_tokens)  # [B, S+K, 2048]
```

### Why Qwen2.5-VL instead of LLaVA-3D

The previous pipeline used LLaVA-3D as VLM, but its vision tower was disabled (only the LLaMA LLM weights were used). Since geometry now bypasses the LLM entirely, LLaVA-3D's spatial ability is wasted. Qwen2.5-VL provides stronger language/vision understanding for the semantic path.

### Key Dimensions

| Symbol | Meaning | Value |
|--------|---------|-------|
| H_dit | DiT / VLM hidden dim | 2048 (Qwen2.5-VL-3B) |
| geom_dim | MapAnything output dim | ~1024 (DINOv2) |
| K | Compressed geometry tokens | 16 (configurable) |
| G | Raw geometry sequence length | variable (~256) |
| N | DiT transformer blocks | 16 |

## Experimental Design

### Baseline

**QwenPI** — Qwen2.5-VL + DiT action head, no geometry at all. This isolates the contribution of 3D information.

### Experiments

Three geometry compression approaches, tested independently:

| Experiment | Config `geom_compression_type` | Description | Extra Params |
|------------|-------------------------------|-------------|-------------|
| Exp A | `pool` | Static prefix: Linear + group pooling, same K tokens at all DiT layers | ~2M |
| Exp B | `per_layer` | Per-layer projection: N separate Linear layers, different projection per DiT block | ~33M |
| Exp C | `qformer` | Q-Former: K learnable queries cross-attend over G tokens | ~26M |

### Ablation Schedule

1. **Exp A (pool)** first — fewest parameters, fastest to validate the concept
2. **Exp C (qformer)** — if pool shows improvement, test adaptive compression
3. **Exp B (per_layer)** — if per-layer specialization matters

### Training Protocol

- Dataset: LIBERO (robot manipulation benchmark)
- Optimizer: AdamW, base lr 1e-5 (VLM), 1e-4 (action head + compressor)
- Scheduler: cosine with min lr 5e-7
- Steps: 100K
- Diffusion: 4 repeated steps per batch, 4 inference steps
- Mixed precision: bf16 for VLM, fp32 for action head
- Freeze: optionally freeze Qwen2.5-VL and MapAnything encoder, train only action head + compressor

### Evaluation

- **LIBERO benchmark**: task success rate across 10 task suites
- Compare against baseline (QwenPI without geometry)
- Metrics: success rate, trajectory smoothness, grasp accuracy

### Config

```yaml
framework:
  name: QwenMapAnythingPI
  qwen_mapanything:
    base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
    mapanything_model_path: /path/to/mapanything
  action_model:
    geom_inject_to_dit: true
    geom_dit_tokens: 16
    geom_compression_type: pool  # pool | per_layer | qformer
```

### Training Command

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_libero_qwen_mapanything_geom_inject.yaml
```

## Implementation Files

| File | Role |
|------|------|
| `starVLA/model/framework/QwenMapAnythingPI.py` | Framework: Qwen2.5-VL + MapAnything + geometry injection |
| `starVLA/model/modules/action_model/geom_compression/static_prefix.py` | Approach A: Static prefix compressor |
| `starVLA/model/modules/action_model/geom_compression/per_layer_projection.py` | Approach B: Per-layer projection compressor |
| `starVLA/model/modules/action_model/geom_compression/qformer_compression.py` | Approach C: Q-Former compressor |
| `starVLA/model/modules/action_model/geom_compression/__init__.py` | Factory: `build_geom_compressor()` |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | DiT action head with geometry injection in cross-attention |
| `starVLA/config/training/starvla_train_libero_qwen_mapanything_geom_inject.yaml` | Training config |

## Expected Outcomes

- **If successful**: 3D geometry injection improves task success rate on LIBERO, especially for tasks requiring precise spatial control (grasping, placing, inserting)
- **If unsuccessful**: geometry features may need richer representation (higher K), or the bottleneck is elsewhere (e.g., action representation, not spatial awareness)
- **Diagnostic signals**: monitor DiT layer-wise attention weights on geometry tokens to verify the model is actually using the 3D information
