# Geometry Direct Injection into DiT Action Head

## Problem

The current `fusion_module()` in `modeling_mapanything_llava3d_vlm.py:253` mean-pools G geometric tokens into 1 averaged vector, destroying all spatial 3D information. By the time geometry reaches the DiT action head, it has been:
1. Collapsed from G spatial tokens to 1 average vector
2. Broadcast identically to every visual patch
3. Passed through the entire LLM (further compressed)

The DiT generates robot trajectories in 3D space — it needs precise geometry the most, but gets the worst version of it.

## Solution: Bypass the LLM

Inject compressed geometric features **directly** into the DiT action head's cross-attention context, so action tokens can attend to both VL semantics (from LLM) and raw 3D geometry (from MapAnything).

```
                     Semantic path (existing)
SigLIP ──> fusion ──> LLM ──> vl_hidden_states ──> DiT cross-attn context ──┐
                                                                              ├──> DiT ──> action
MapAnything ──> compress to K tokens ──> DiT cross-attn context ────────────┘
                     Geometric path (NEW)
```

At each DiT layer, the cross-attention context becomes:
```
cross_context = concat(vl_embs_list[layer_idx], geom_tokens)  # [B, S+K, H_dit]
```

---

## Dimension Reference

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| B | Batch size | varies |
| G | Geometric sequence length (MapAnything output) | ~256-1024 |
| K | Compressed tokens (configurable) | 16 (default) |
| geom_dim | MapAnything feature dimension | ~1024 |
| H_dit | DiT hidden / cross-attention dimension | 4096 (= vl_hidden_dim) |
| N | Number of DiT transformer blocks | 16-36 |
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
  │     [B, G, geom_dim] ──> [B, G, H_dit]
  │
  ├── Step 2: Group pooling (G -> K tokens)
  │     Split G into K groups of size G//K, mean-pool each group
  │     [B, G, H_dit] ──reshape──> [B, K, G//K, H_dit] ──mean(dim=2)──> [B, K, H_dit]
  │
  └── Output: [B, K, H_dit]   (same tokens used at EVERY DiT layer)
```

### How It's Used in the DiT

```python
geom_tokens = compressor(raw_geom_feats)        # [B, K, H_dit]  (computed once)

for layer_idx, layer in enumerate(dit_blocks):
    context = cat([vl_embs_list[layer_idx], geom_tokens], dim=1)  # [B, S+K, H_dit]
    hidden_states = layer(hidden_states, encoder_hidden_states=context)
```

### Properties

| Property | Value |
|----------|-------|
| Parameters added | `geom_dim * H_dit` (one linear layer, ~4M for 1024*4096) |
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
  │     │     [B, G, geom_dim] ──> [B, G, H_dit]
  │     │
  │     └── Step 2: Group pooling (same as Approach A)
  │           [B, G, H_dit] ──> [B, K, H_dit]
  │
  └── Each DiT layer gets a DIFFERENT projection of the same geometry
```

### How It's Used in the DiT

```python
for layer_idx, layer in enumerate(dit_blocks):
    geom_tokens = compressor(raw_geom_feats, layer_idx)  # [B, K, H_dit] (different per layer!)
    context = cat([vl_embs_list[layer_idx], geom_tokens], dim=1)
    hidden_states = layer(hidden_states, encoder_hidden_states=context)
```

### Properties

| Property | Value |
|----------|-------|
| Parameters added | `N * geom_dim * H_dit` (N linear layers, ~64M for 16*1024*4096) |
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
  │     [B, G, geom_dim] ──> [B, G, H_dit]
  │
  ├── Step 2: K learnable query tokens (nn.Parameter)
  │     self.geom_queries = nn.Parameter(randn(1, K, H_dit))
  │     Expand to batch: [1, K, H_dit] ──> [B, K, H_dit]
  │
  ├── Step 3: Cross-attention
  │     query = learned queries    [B, K, H_dit]  ←── "what to extract"
  │     key   = projected geometry [B, G, H_dit]  ←── "what's available"
  │     value = projected geometry [B, G, H_dit]  ←── "what gets read"
  │
  │     For each query q_i:
  │       scores_j = dot(q_i, geom_j) / sqrt(H_dit)   for j=0..G-1
  │       weights = softmax(scores)                    [K, G]
  │       output_i = sum(weights_j * geom_j)           weighted combination
  │
  │     Output: [B, K, H_dit]
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
| Parameters added | `geom_dim * H_dit + K * H_dit + 3 * H_dit * H_dit` (~52M) |
| Sequence length increase | +K per DiT layer cross-attention |
| Adaptive to input | Yes (attention weights change per scene) |
| Per-layer specialization | No (same compressed tokens at every layer) |

---

## Comparison Table

| | Approach A (Static) | Approach B (Per-Layer) | Approach C (Q-Former) |
|---|---|---|---|
| Compression method | Group pooling | Group pooling | Learned cross-attention |
| Parameters | ~4M | ~64M (16 layers) | ~52M |
| Input-adaptive | No | No | Yes |
| Per-layer specialized | No | Yes | No |
| Complexity | Very low | Low | Medium |
| Training data needed | Least | More (more params) | More (attention must specialize) |
| Best when | Validating the concept | Layers need different views | Rich/varied geometry |

---

## Configuration

```yaml
framework:
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

## Files Modified

| File | Change |
|------|--------|
| `starVLA/model/modules/action_model/geom_compression/__init__.py` | Factory `build_geom_compressor()` |
| `starVLA/model/modules/action_model/geom_compression/static_prefix.py` | Approach A |
| `starVLA/model/modules/action_model/geom_compression/per_layer_projection.py` | Approach B |
| `starVLA/model/modules/action_model/geom_compression/qformer_compression.py` | Approach C |
| `starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py` | Return raw geom features before fusion |
| `starVLA/model/framework/MapAnythingLlava3DPI.py` | Instantiate compressor, pass geom to action head |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | Accept `geom_tokens` in cross-attention |
