# Feature Fusion Strategies for MapAnythingLlava3DPI

## Context: What Are We Fusing?

The `MapAnythingLlava3DPI` framework has three distinct information sources that must be combined before the DiT action head can predict robot trajectories:

| Modality | Source module | Output shape | What it encodes |
|---|---|---|---|
| **2D semantic vision** | SigLIP vision tower | `[B, V*256, H]` | Texture, color, object category, semantic content |
| **3D geometry** | MapAnythingWrapper | `[B, G, geom_dim]` | Depth per patch, surface normals, multi-view 3D correspondences |
| **Language instruction** | Tokenizer → LLM embedding | `[B, L, H]` | Task intent, object identity, spatial relations |

- `B` = batch size (number of samples processed in parallel)
- `V` = number of camera views (e.g. 2)
- `256` = SigLIP patch tokens per image (16×16 grid from 224px / 14px patch size)
- `G` = geometric sequence length from MapAnything (likely differs from V*256)
- `H` = LLM hidden dimension (e.g. 4096)
- `L` = language sequence length

---

## The Core Problem with the Current Approach

All fusion currently happens in `fusion_module()` at `modeling_mapanything_llava3d_vlm.py:253`:

```python
def fusion_module(self, geometric_features, vision_features):
    geometric_features = self.geometric_projector(geometric_features)   # [B, G, H]
    geometric_global = geometric_features.mean(dim=1, keepdim=True)     # [B, 1, H]  ← POOL TO 1
    geometric_broadcast = geometric_global.expand(..., V*256, ...)       # [B, V*256, H] ← SAME FOR ALL
    fused = torch.cat([vision_features, geometric_broadcast], dim=-1)   # [B, V*256, 2H]
    return self.fusion_projector(fused)                                  # [B, V*256, H]
```

**The structural problems:**
1. Mean pooling collapses G spatially distinct geometric tokens → 1 averaged scalar vector. All per-location depth, normal, and correspondence information is destroyed.
2. Every visual patch (the cup patch, the table patch, the wall patch) receives the identical averaged geometric vector via broadcast.
3. Language has no role here — geometry is pooled identically regardless of the task instruction.
4. G ≠ V*256 (MapAnything and SigLIP use different spatial resolutions), so naive token-wise alignment is not straightforward.

The code has a `# TODO: revisit geometric pooling strategy` comment at this exact location, acknowledging it is a placeholder.

---

## What Does the Action Head Actually Need?

The DiT flow-matching action head generates a robot trajectory — a sequence of 3D poses or joint angles over time. Its cross-attention context comes from the LLM's hidden states, which encode the blended language + visual + (pooled) geometric signal.

Thinking about what each modality contributes to trajectory generation:
- **Language**: *what* to do, *which* object to manipulate
- **Vision**: *where* visually in the image the target is, *what it looks like*
- **Geometry**: *how far* the target is in metric 3D space, *what angle* to approach, *what 3D shape* to grasp

This reveals a key insight: **geometry is most valuable at the action prediction step, not at the semantic reasoning step.** The LLM handles semantic understanding. The DiT generates the 3D trajectory. Geometry should be closest to the DiT.

---

## Strategy 1 — Spatial Cross-Attention Fusion

**Core idea:** Replace the mean pool + broadcast + linear with a proper cross-attention layer. Each visual patch token acts as a query and attends over all G geometric tokens to selectively pull the most relevant 3D information for its spatial location.

```
vision  [B, V*256, H]  ←── queries ──┐
                                      ├── nn.MultiheadAttention → [B, V*256, H]
geom    [B, G, H]      ←── keys/vals ─┘
```

Concretely replacing `fusion_module()`:

```python
class SpatialCrossAttnFusion(nn.Module):
    def __init__(self, hidden_dim, geom_dim, num_heads=8):
        self.geom_proj = nn.Linear(geom_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, vision_feats, geom_feats):
        geom = self.geom_proj(geom_feats)                          # [B, G, H]
        attended, _ = self.cross_attn(
            query=vision_feats, key=geom, value=geom               # vision queries geom
        )                                                           # [B, V*256, H]
        return self.output_proj(torch.cat([vision_feats, attended], dim=-1))  # [B, V*256, H]
```

Output shape `[B, V*256, H]` is identical to the current `fusion_module()` output — nothing downstream changes.

**Pros:**
- Directly fixes the core problem with minimal downstream impact
- No sequence length change; no new tokens in the LLM
- Each visual patch selectively attends to its most relevant geometry
- Does not require G to equal V*256 (cross-attention handles variable-length keys)

**Cons:**
- Adds a cross-attention module (~4M params for H=4096, num_heads=8)
- Language still does not guide geometry selection at this stage
- Geometry still enters the LLM pre-fused, not as independently addressable tokens

**Literature precedent:** DETR encoder, Perceiver, Flamingo cross-attention layers, InternVL multi-scale fusion.

**Implementation effort:** Low. Only `fusion_module()` changes. Add config flag `fusion_type: cross_attn`.

---

## Strategy 2 — Geometry as Separate Prefix Tokens in the LLM

**Core idea:** Compress geometry to K tokens (via pooling, linear projection, or a small Q-Former), then prepend them as a prefix to the LLM's input sequence. The LLM's own self-attention then lets language, visual, and geometric tokens freely attend to each other.

```
LLM input sequence:
[geom_tok_0, ..., geom_tok_K,        ← K compressed geometric prefix tokens (NEW)
 <img_0>, <img_1>, ..., <img_511>,   ← V*256 visual tokens (as now)
 lang_tok_0, ..., lang_tok_N]        ← language tokens (as now)
```

Language token at position `lang_tok_i` can now directly attend to `geom_tok_j` via LLM self-attention. The language can ask: "which geometric region corresponds to the cup I need to pick up?" Visual tokens can also cross-attend to geometric tokens without the early pooling bottleneck.

K is a tunable memory budget:
- K=8: minimal sequence length increase (~1.5%), coarse geometry
- K=16: balanced (3%), recommended starting point
- K=32: richer geometry (6%), higher memory cost

**Pros:**
- Language can now guide geometric attention (language tokens attend to geom prefix)
- Geometry is independently addressable, not pre-fused with vision
- K is a simple hyperparameter to tune
- Relatively small implementation change (modify LLM input construction)

**Cons:**
- Sequence length increases by K → more memory and compute at every LLM layer
- K-token compression still loses some spatial detail (better than 1 token, but not perfect)
- Need a compression module (simple: `nn.Linear(geom_dim, H)` + pooling groups; advanced: Q-Former)

**Literature precedent:** InternVL-2 (multiple encoder tokens concatenated before LLM), Cambrian-1, mPLUG-Owl3.

**Implementation effort:** Low–Medium. Modify `build_mapanythingllava3d_inputs()` to prepend geometric tokens. Add config key `geom_prefix_tokens: 16`.

---

## Strategy 3 — Language-Guided Geometric Attention

**Core idea:** Use the language instruction to selectively weight geometric features *before* fusing them with vision. "Pick up the red cup" should amplify the cup's geometry and suppress the table's geometry. This directly solves the problem that language cannot guide geometric selection in the current architecture.

```
instruction text
    → tokenize → LLM embedding → mean pool → lang_emb [B, 1, H]
                                                    ↓  (as query)
geom tokens [B, G, H]  ←── cross-attention ──── lang_emb
                                                    ↓
lang_conditioned_geom [B, G, H]   ← geometry re-weighted by task relevance
                                                    ↓
then fuse with vision (via Strategy 1 cross-attention or concat)
```

The instruction embedding acts as a task-specific attention query over the full geometric sequence. Regions geometrically relevant to the instruction receive higher attention weights; irrelevant regions are suppressed.

**Pros:**
- Directly addresses the language–geometry alignment problem
- Task-relevant geometric features are amplified rather than uniformly averaged
- Most principled solution to the "language cannot guide geometry" problem

**Cons:**
- Requires computing a language embedding separately (before or alongside the LLM forward pass)
- Adds a cross-attention module between language and geometry
- Encoding quality of the language embedding matters — using just the first LLM hidden state may be insufficient for complex instructions

**Literature precedent:** GroundingDINO (text-guided visual feature selection), GLIP, RegionCLIP, CLIP-guided attention.

**Implementation effort:** Medium. Requires extracting a language embedding before the geometry–vision fusion step. Best combined with Strategy 1.

---

## Strategy 4 — Geometry Directly Into the DiT Action Head (Bypass the LLM)

**Core idea:** The LLM does not need raw geometry for semantic reasoning — it needs to understand the task. The DiT needs 3D structure for trajectory generation. So send geometric features directly to the DiT as additional cross-attention context, completely bypassing the LLM bottleneck.

```
SigLIP → [B, V*256, H] → LLM → vl_hidden_states → DiT cross-attention ─────────────────────┐
                                                                                              ├→ DiT → action
MapAnything → [B, G, geom_dim] → geometric_projector → [B, K, H_dit]  → DiT cross-attention ─┘
```

In the DiT, each transformer block cross-attends to a concatenated context:

```python
# In LayerwiseFM_ActionHeader, each DiT block i:
context = torch.cat([vl_embs_list[i], geom_tokens], dim=1)  # [B, L+K, H_dit]
# DiT cross-attention: action tokens attend to both VL semantics AND raw geometry
```

**This infrastructure already exists.** The `use_concat_cross_context: true` config flag in the ablation YAMLs was designed for exactly this purpose — concatenating additional context to the DiT cross-attention keys/values. The flag exists in `LayerwiseFM_ActionHeader` and in the config files `ab_b_concat_cross_geometric.yaml` and `ab_b_concat_cross_geometric_alg1.yaml`.

**Pros:**
- Architectural separation of concerns: LLM handles semantics, DiT handles 3D trajectory
- Geometry arrives at the action head without LLM compression losses
- Most directly targeted at the action prediction quality
- The existing `use_concat_cross_context` infrastructure means this is partially scaffolded already

**Cons:**
- Geometry is not modulated by language (language affects geometry only indirectly via the LLM path)
- Two separate information pathways to maintain and tune
- K needs to be chosen (compression from G to K tokens needed before DiT injection)

**Literature precedent:** GR00T N1.5 dual-system architecture (System 2 semantic VLM, System 1 fast action), Octo cross-attention from action tokens to observation tokens.

**Implementation effort:** Low. The scaffold is already in place. Wire `geom_tokens` into the existing concat-context path in `LayerwiseFM_ActionHeader.forward()`.

---

## Strategy 5 — 3D Positional Encoding Injection

**Core idea:** Instead of treating geometry as a separate feature branch to be fused, use MapAnything's depth output to compute a metric 3D position `(X, Y, Z)` for each visual patch, then add a learned 3D positional embedding to each visual token before it enters the LLM.

```
depth map [B, 16, 16] extracted from MapAnything
  + camera intrinsics K (focal length, principal point)
  → backproject to 3D: X = (u - cx) * depth / fx,  Y = (v - cy) * depth / fy,  Z = depth
  → per-patch 3D coords [B, 256, 3]
  → 3D positional encoder (MLP or sinusoidal) → [B, 256, H]
  → visual_tokens += 3D_pos_embedding   ← additive, no new tokens
```

The LLM sees visual tokens that carry both semantic content (what the patch looks like) AND metric 3D position (exactly where in space it is). No new tokens are added; only the embedding values change.

**Pros:**
- Zero sequence length increase
- No new modules except a small positional encoder (MLP ~1M params)
- LLM becomes inherently 3D-position-aware
- Language can naturally attend to spatially-indexed visual tokens
- Camera intrinsics are already passed through the pipeline (`intrinsic` tensor in `build_mapanythingllava3d_inputs()`)

**Cons:**
- Only injects position (X, Y, Z), not surface normals or multi-view geometric structure
- Requires reliable depth estimates from MapAnything for every patch
- LLM was not pretrained with 3D positional encodings — there may be an adaptation cost at the start of training
- If MapAnything's depth is noisy, the positional embeddings are noisy too

**Literature precedent:** 3D-LLM, SpatialBot, EmbodiedSAM, PointLLM, LEO (embodied agent with 3D scene understanding).

**Implementation effort:** Low. Add a `ThreeDPositionalEncoder` module; modify `get_image_features()` to extract depth and add the embedding before returning.

---

## Strategy 6 — Hierarchical Two-Stage Fusion

**Core idea:** Fuse at two levels of granularity, matching how spatial cognition works:
- Stage A (local, before LLM): each visual patch fused with its spatially aligned geometric patch → captures local 3D-semantic detail
- Stage B (global, in LLM): a K-token global geometric summary prepended as prefix → gives LLM a scene-wide geometric overview that language can attend to

```
Stage A — per-patch local fusion:
  vision [B, V*256, H]  \
                         ├─ cross-attn or gated fusion → local_fused [B, V*256, H]
  geom   [B, G, H]      /

Stage B — global prefix in LLM:
  geom [B, G, H] → pool to K tokens → [B, K, H]    ← global summary
  LLM input: [global_geom_prefix_K, local_fused_visual_V*256, language_L]
```

The combination gives the LLM access to:
- Local: each patch's specific 3D context (from Stage A)
- Global: a scene-wide geometric summary that language tokens can attend to directly (from Stage B)

**Pros:**
- Captures both local spatial correspondence and global scene geometry
- Language can attend to global geometry (Stage B prefix) and to locally-fused visual tokens (Stage A output)
- Matches human perceptual hierarchy (global layout → local detail)

**Cons:**
- Two separate fusion stages to design, implement, and tune
- Sequence length increases by K (from Stage B prefix)
- Higher implementation complexity

**Literature precedent:** InternVL-2 multi-scale visual encoding, LLaVA-HR high-resolution patch fusion, Uni3D.

**Implementation effort:** High. Requires implementing both stages and integrating them into the existing pipeline without breaking other frameworks.

---

## Strategy 7 — Gated Geometric Modulation (Quickest Meaningful Fix)

**Core idea:** Keep the overall structure but replace the mean pool with a soft spatial gate. Geometry is spatially interpolated to match the vision grid, then used to gate each visual token — amplifying patches where geometry is confidently informative (near an object), suppressing patches where it is not.

```python
# Align geometry to vision spatial grid via bilinear interpolation
# (requires G to be reshapable into H_g × W_g, true if MapAnything output is a spatial feature map)
geom_spatial = geom.transpose(1,2).reshape(B, geom_dim, H_g, W_g)
geom_aligned = F.interpolate(geom_spatial, size=(16, 16), mode='bilinear', align_corners=False)
geom_aligned = geom_aligned.flatten(2).transpose(1, 2)    # [B, 256, geom_dim]
geom_proj    = self.geometric_projector(geom_aligned)      # [B, 256, H]

# Soft gate: how much does geometry modulate each patch?
gate = torch.sigmoid(self.gate_linear(geom_proj))          # [B, 256, 1]
fused = vision_feats * gate + geom_proj * (1 - gate)       # [B, 256, H]
```

**Pros:**
- Minimal parameters (one additional linear layer for the gate)
- Preserves spatial structure if G can be reshaped into a grid
- No sequence length change
- Easiest to implement and test quickly

**Cons:**
- Requires MapAnything output to be a spatial grid (H_g × W_g) to interpolate; may not always hold
- Bilinear interpolation between grids of different sizes is lossy
- No language guidance
- Gating is learned globally, not task-conditioned

**Literature precedent:** Feature-wise linear modulation (FiLM), gated attention in visual question answering.

**Implementation effort:** Very low. Only `fusion_module()` changes.

---

## Comparative Summary

| Strategy | Preserves spatial geometry | Language guides geometry | Sequence length change | Implementation effort | Scaffolding already present? |
|---|---|---|---|---|---|
| 1. Spatial cross-attention | ✅ Yes | ❌ No | None | Low | No |
| 2. Geometry as LLM prefix | ✅ Partial (K tokens) | ✅ Yes (via LLM attn) | +K tokens | Low–Medium | No |
| 3. Language-guided geom attn | ✅ Yes | ✅ Yes (explicit) | None | Medium | No |
| 4. Geometry → DiT directly | ✅ Yes | ❌ Indirect | None | Low | **Yes** (`use_concat_cross_context`) |
| 5. 3D positional encoding | ✅ Position only | ✅ Yes (via LLM attn) | None | Low | Partial (intrinsics already passed) |
| 6. Hierarchical two-stage | ✅ Yes (both levels) | ✅ Yes | +K tokens | High | No |
| 7. Gated modulation | ✅ If grids align | ❌ No | None | Very low | No |

---

## Recommended Path Forward

### Phase 1 — Validate geometry actually helps (cheapest experiments)

Run Strategy 7 (gated modulation) first. It is the cheapest change that preserves spatial structure and will confirm whether geometry adds signal over the current mean-pool baseline. If gated modulation does not improve over no-geometry (`ab_a_pure_cross`), then the issue is deeper than the fusion mechanism.

### Phase 2 — Fix the core structural problem

Implement Strategy 1 (spatial cross-attention fusion). This is the principled replacement for the mean pool. Output shape is identical, so nothing downstream changes. Add a config flag `fusion_type: cross_attn | mean_pool` to allow A/B comparison.

### Phase 3 — Give geometry access to the action head

Implement Strategy 4 (geometry directly into DiT). The `use_concat_cross_context` flag is already wired in the DiT code and the ablation configs. This is the highest-leverage change for action quality because it gives the action head direct 3D access without the LLM as a bottleneck. Run `ab_b_concat_cross_geometric.yaml` with actual geometry wired in.

### Phase 4 — Close the language–geometry gap

Combine Strategy 2 (geometry prefix tokens) or Strategy 3 (language-guided geometric attention) to let language modulate which geometric features matter. This is where the most research novelty lies.

### Ideal end-state architecture

```
                ┌─────────────────────────────────────────────────────┐
                │                  SEMANTIC PATH                       │
images ──► SigLIP ──► spatial cross-attn with geom (Strategy 1) ──► LLM ──► vl_hidden_states ──►┐
                │                  (3D-aware visual tokens enter LLM)  │                          │
                └─────────────────────────────────────────────────────┘                          │
                                                                                                  ├──► DiT ──► action
                ┌─────────────────────────────────────────────────────┐                          │
                │                  GEOMETRIC PATH                      │                          │
images ──► MapAnything ──► K-token compression ──────────────────────────────────────────────────►┘
                │         (raw 3D directly to DiT, Strategy 4)         │
                └─────────────────────────────────────────────────────┘
```

- Semantic path: geometry modulates LLM reasoning (the LLM sees 3D-aware visual tokens)
- Geometric path: geometry directly informs trajectory generation (DiT has raw 3D access)

The two paths are complementary, not redundant. Language influences the semantic path naturally via LLM self-attention. The geometric path bypasses the LLM to ensure 3D precision is not lost in the language modelling process.

---

## Key Files to Modify per Strategy

| Strategy | Primary file | Function to change |
|---|---|---|
| 1, 3, 7 | `starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py` | `fusion_module()` (line 253) |
| 2 | `starVLA/model/modules/vlm/MapAnythingLlava3D.py` | `build_mapanythingllava3d_inputs()` |
| 4 | `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | `forward()` — wire geom into concat context |
| 4 | `starVLA/model/framework/MapAnythingLlava3DPI.py` | `forward()` / `predict_action()` — pass geom to action model |
| 5 | `starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py` | `get_image_features()` — add 3D pos encoding |
| 6 | All of the above | Both Stage A and Stage B |
