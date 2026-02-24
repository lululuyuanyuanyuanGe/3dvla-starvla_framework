# Geometry Direct Injection into DiT Action Head

## Problem

The current `fusion_module()` mean-pools G geometric tokens into 1 averaged vector, destroying all spatial 3D information. The DiT generates robot trajectories in 3D space — it needs precise geometry the most, but gets the worst version of it.

## Solution: Bypass the LLM

Inject compressed geometric features **directly** into the DiT action head's cross-attention context, so action tokens can attend to both VL semantics (from LLM) and raw 3D geometry (from MapAnything).

### Architecture (Qwen2.5-VL + MapAnything)

```
                     Semantic path
Qwen2.5-VL (images + instruction) ──> multi-layer hidden states ──> DiT cross-attn context ──┐
                                                                                               ├──> DiT ──> action
MapAnything (standalone) ──> compress G→K tokens ──> DiT cross-attn context ──────────────────┘
                     Geometric path (direct injection)
```

At each DiT layer, the cross-attention context becomes:
```
cross_context = concat(vl_embs_list[layer_idx], geom_tokens)  # [B, S+K, H_dit]
```

### Framework

**`QwenMapAnythingPI`** — Qwen2.5-VL provides the semantic path (language + vision understanding), MapAnything runs standalone as a pure geometric encoder (DINOv2 + multi-view transformer), and the 3D features are injected directly into the DiT action head.

---

## Dimension Reference

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| B | Batch size | varies |
| G | Geometric sequence length (MapAnything output) | ~256 |
| K | Compressed tokens (configurable) | 16 (default) |
| geom_dim | MapAnything feature dimension | ~1024 |
| H_dit | DiT hidden / cross-attention dimension | 2048 (Qwen2.5-VL-3B) |
| N | Number of DiT transformer blocks | 16 |
| S | VL sequence length (lang + visual tokens) | ~512-1024 |

---

## Approach A: Static Prefix Compression

**File:** `starVLA/model/modules/action_model/geom_compression/static_prefix.py`
**Config:** `geom_compression_type: pool`

### Mechanism

```
MapAnything output: [B, G, geom_dim]
  │
  ├── Step 1: Linear projection
  │     nn.Linear(geom_dim, H_dit)
  │     [B, G, 1024] ──> [B, G, 2048]
  │
  ├── Step 2: Group pooling (G -> K tokens)
  │     Split G into K groups of size G//K, mean-pool each group
  │     [B, G, 2048] ──reshape──> [B, K, G//K, 2048] ──mean(dim=2)──> [B, K, 2048]
  │
  └── Output: [B, K, 2048]   (same tokens used at EVERY DiT layer)
```

### How It's Used in the DiT

```python
geom_tokens = compressor(raw_geom_feats)        # [B, K, 2048]  (computed once)

for layer_idx, layer in enumerate(dit_blocks):
    context = cat([vl_embs_list[layer_idx], geom_tokens], dim=1)  # [B, S+K, 2048]
    hidden_states = layer(hidden_states, encoder_hidden_states=context)
```

### Properties

| Property | Value |
|----------|-------|
| Parameters added | `geom_dim * H_dit` (one linear layer, ~2M for 1024*2048) |
| Sequence length increase | +K per DiT layer cross-attention |
| Adaptive to input | No (fixed group boundaries) |
| Per-layer specialization | No (same tokens at every layer) |

---

## Approach B: Per-Layer Projection

**File:** `starVLA/model/modules/action_model/geom_compression/per_layer_projection.py`
**Config:** `geom_compression_type: per_layer`

### Mechanism

```
MapAnything output: [B, G, geom_dim]
  │
  ├── For DiT layer i:
  │     ├── Step 1: Layer-specific linear projection
  │     │     self.projectors[i] = nn.Linear(geom_dim, H_dit)   # different weights per layer!
  │     │     [B, G, 1024] ──> [B, G, 2048]
  │     │
  │     └── Step 2: Group pooling (same as Approach A)
  │           [B, G, 2048] ──> [B, K, 2048]
  │
  └── Each DiT layer gets a DIFFERENT projection of the same geometry
```

### How It's Used in the DiT

```python
for layer_idx, layer in enumerate(dit_blocks):
    geom_tokens = compressor(raw_geom_feats, layer_idx)  # [B, K, 2048] (different per layer!)
    context = cat([vl_embs_list[layer_idx], geom_tokens], dim=1)
    hidden_states = layer(hidden_states, encoder_hidden_states=context)
```

### Properties

| Property | Value |
|----------|-------|
| Parameters added | `N * geom_dim * H_dit` (N linear layers, ~33M for 16*1024*2048) |
| Sequence length increase | +K per DiT layer cross-attention |
| Adaptive to input | No |
| Per-layer specialization | Yes (each layer can extract different geometric aspects) |

---

## Approach C: Q-Former Compression

**File:** `starVLA/model/modules/action_model/geom_compression/qformer_compression.py`
**Config:** `geom_compression_type: qformer`

### Mechanism

```
MapAnything output: [B, G, geom_dim]
  │
  ├── Step 1: Linear projection
  │     nn.Linear(geom_dim, H_dit)
  │     [B, G, 1024] ──> [B, G, 2048]
  │
  ├── Step 2: K learnable query tokens (nn.Parameter)
  │     self.geom_queries = nn.Parameter(randn(1, K, H_dit))
  │     Expand to batch: [1, K, 2048] ──> [B, K, 2048]
  │
  ├── Step 3: Cross-attention
  │     query = learned queries    [B, K, 2048]  ←── "what to extract"
  │     key   = projected geometry [B, G, 2048]  ←── "what's available"
  │     value = projected geometry [B, G, 2048]  ←── "what gets read"
  │
  │     For each query q_i:
  │       scores_j = dot(q_i, geom_j) / sqrt(2048)   for j=0..G-1
  │       weights = softmax(scores)                    [K, G]
  │       output_i = sum(weights_j * geom_j)           weighted combination
  │
  │     Output: [B, K, 2048]
  │
  └── Same K tokens used at every DiT layer (same as Approach A)
```

### Key Difference from A/B

Group pooling (A/B) uses **fixed** groups — token positions 0-15 always go to group 0.
Q-Former (C) uses **learned, input-dependent** weights — each query can attend to
ANY combination of the G tokens, and the attention pattern changes per scene.

```
Approach A (fixed groups):
  [g0 g1 g2 g3 | g4 g5 g6 g7]  →  [avg(g0-g3), avg(g4-g7)]  (rigid boundaries)

Approach C (learned queries):
  Query 0: attends 35% to g2, 28% to g6, 20% to g3, ...  (adaptive, scene-dependent)
  Query 1: attends 40% to g0, 30% to g1, ...              (different specialization)
```

### Properties

| Property | Value |
|----------|-------|
| Parameters added | `geom_dim * H_dit + K * H_dit + 3 * H_dit * H_dit` (~26M) |
| Sequence length increase | +K per DiT layer cross-attention |
| Adaptive to input | Yes (attention weights change per scene) |
| Per-layer specialization | No (same compressed tokens at every layer) |

---

## Comparison Table

| | Approach A (Static) | Approach B (Per-Layer) | Approach C (Q-Former) |
|---|---|---|---|
| Compression method | Group pooling | Group pooling | Learned cross-attention |
| Parameters | ~2M | ~33M (16 layers) | ~26M |
| Input-adaptive | No | No | Yes |
| Per-layer specialized | No | Yes | No |
| Complexity | Very low | Low | Medium |
| Training data needed | Least | More (more params) | More (attention must specialize) |
| Best when | Validating the concept | Layers need different views | Rich/varied geometry |

---

## Configuration

```yaml
framework:
  name: QwenMapAnythingPI
  qwen_mapanything:
    base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
    mapanything_model_path: /path/to/mapanything

  action_model:
    # Enable geometry injection into DiT
    geom_inject_to_dit: true

    # Number of compressed geometric tokens (K)
    geom_dit_tokens: 16

    # Compression approach: pool | per_layer | qformer
    geom_compression_type: pool

    # (qformer only) Number of attention heads
    geom_qformer_heads: 8
```

## Files

| File | Role |
|------|------|
| `starVLA/model/framework/QwenMapAnythingPI.py` | Framework: Qwen2.5-VL + standalone MapAnything + geometry injection |
| `starVLA/model/modules/action_model/geom_compression/__init__.py` | Factory `build_geom_compressor()` |
| `starVLA/model/modules/action_model/geom_compression/static_prefix.py` | Approach A |
| `starVLA/model/modules/action_model/geom_compression/per_layer_projection.py` | Approach B |
| `starVLA/model/modules/action_model/geom_compression/qformer_compression.py` | Approach C |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | DiT action head: accepts `geom_tokens` in cross-attention |
| `starVLA/config/training/starvla_train_libero_qwen_mapanything_geom_inject.yaml` | Training config |
