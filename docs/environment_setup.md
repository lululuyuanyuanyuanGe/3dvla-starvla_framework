# Environment Setup: QwenMapAnythingPI

Fresh conda environment for running and training the QwenMapAnythingPI framework (Qwen2.5-VL + MapAnything + direct geometry injection to DiT action head).

## Prerequisites

- Linux with NVIDIA GPU (12-16GB+ VRAM for smoke test, multi-GPU for training)
- CUDA toolkit 12.x installed (`nvcc -V` to check)
- conda or miniconda

## Step 1: Create conda environment

```bash
conda create -n qwen_mapanything python=3.10 -y
conda activate qwen_mapanything
```

## Step 2: Install PyTorch

Match your CUDA version. Check with `nvcc -V`.

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Step 3: Install StarVLA dependencies

```bash
cd /2025233147/zzq/SpatialVLA_llava3d/3dvla-starvla_framework
pip install -r requirements.txt
```

## Step 4: Install FlashAttention2

```bash
pip install flash-attn --no-build-isolation
```

`flash-attn==2.7.4.post1` is verified to work with CUDA 12.0 and 12.4. If build fails, ensure `nvcc -V` CUDA version matches your PyTorch CUDA version.

## Step 5: Install MapAnything (git submodule)

```bash
git submodule update --init --recursive
pip install -e starVLA/mapanything_llava3d/model/map-anything/
```

This automatically installs `uniception` (MapAnything's multi-view transformer dependency).

## Step 6: Install StarVLA

```bash
pip install -e .
```

## Step 7: Verify installation

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
import diffusers; print(f'Diffusers: {diffusers.__version__}')
import flash_attn; print(f'FlashAttn: {flash_attn.__version__}')
from mapanything.models.mapanything.model import MapAnything; print('MapAnything: OK')
from uniception.models.info_sharing.base import MultiViewTransformerInput; print('Uniception: OK')
from qwen_vl_utils import process_vision_info; print('qwen-vl-utils: OK')
print('All dependencies OK')
"
```

## Step 8: Pretrained models

```bash
# Qwen2.5-VL-3B — download from HuggingFace if not present
ls ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct || \
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')"

# MapAnything weights — already at:
ls /2025233147/zzq/mapAnythingLlava3dPi0.5/model_zoo/mapanything
```

## Step 9: Smoke test

```bash
# Test each compression approach (needs 1 GPU)
python starVLA/model/framework/QwenMapAnythingPI.py --geom_compression_type pool
python starVLA/model/framework/QwenMapAnythingPI.py --geom_compression_type per_layer
python starVLA/model/framework/QwenMapAnythingPI.py --geom_compression_type qformer
```

Each run builds the full model, runs a forward pass with fake data, checks gradients, and runs inference.

## Step 10: Launch training

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_libero_qwen_mapanything_geom_inject.yaml \
  --framework.action_model.geom_compression_type pool \
  --run_id exp_geom_pool
```

Switch `geom_compression_type` to `per_layer` or `qformer` for the other approaches.

## Dependency Summary

| Package | Version | Source |
|---------|---------|--------|
| torch | (match CUDA) | PyPI cu121/cu124 |
| transformers | 4.57.0 | requirements.txt |
| diffusers | latest | requirements.txt |
| accelerate | 1.5.2 | requirements.txt |
| deepspeed | 0.16.9 | requirements.txt |
| flash-attn | 2.7.4.post1 | separate pip install |
| mapanything | 1.1 | git submodule editable |
| uniception | 0.1.6 | auto-installed by mapanything |
| qwen-vl-utils | latest | requirements.txt |
| omegaconf | latest | requirements.txt |
| wandb | latest | requirements.txt |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| flash-attn build fails | `nvcc -V` CUDA must match PyTorch CUDA. Try pinning: `pip install flash-attn==2.7.4.post1 --no-build-isolation` |
| MapAnything import error | Run `git submodule update --init --recursive` then `pip install -e starVLA/mapanything_llava3d/model/map-anything/` |
| OOM on smoke test | Reduce tokens: `--framework.action_model.geom_dit_tokens 8` |
| uniception not found | Should be auto-installed by mapanything. Manual: `pip install uniception>=0.1.6` |
