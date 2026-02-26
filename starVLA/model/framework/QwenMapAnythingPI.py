"""
QwenMapAnythingPI Framework
Combines Qwen2.5-VL (semantic path) + standalone MapAnything (geometric path)
with direct geometry injection into the DiT action head.

Architecture:
  Images + Instruction --> Qwen2.5-VL --> multi-layer hidden states --> DiT cross-attn (semantic)
                                                                          |
  Images --> MapAnything (standalone) --> compress G->K tokens ----------> DiT cross-attn (geometry)
                                                                          |
                                                                          v
                                                                         DiT --> action
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    get_action_model,
    LayerwiseFlowmatchingActionHead,
)
from starVLA.model.modules.action_model.geom_compression import build_geom_compressor
from starVLA.model.modules.action_model.geom_compression.per_layer_projection import PerLayerProjectionCompressor
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.modules.geometric import MapAnythingWrapper

logger = initialize_overwatch(__name__)


# DINOv2 ImageNet normalization constants
_DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _pil_images_to_mapanything_tensor(batch_images, target_size=(224, 224), device="cpu"):
    """Convert a batch of PIL image lists to a [B, V, 3, H, W] tensor for MapAnything.

    MapAnything internally applies DINOv2 normalization (data_norm_type=["dinov2"]),
    so we just need to provide images in [0, 1] float range.

    Args:
        batch_images: list of list of PIL.Image (B samples, V views each)
        target_size: (H, W) to resize to
        device: target device
    Returns:
        pixel_values: [B, V, 3, H, W] float tensor in [0, 1]
    """
    import torchvision.transforms.functional as TF

    batch_tensors = []
    for images_per_sample in batch_images:
        view_tensors = []
        for img in images_per_sample:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            img = img.convert("RGB")
            if img.size != (target_size[1], target_size[0]):
                img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
            view_tensors.append(t)
        batch_tensors.append(torch.stack(view_tensors, dim=0))  # [V, 3, H, W]

    pixel_values = torch.stack(batch_tensors, dim=0)  # [B, V, 3, H, W]
    return pixel_values.to(device=device)


@FRAMEWORK_REGISTRY.register("QwenMapAnythingPI")
class QwenMapAnything_PI(baseframework):
    """
    Qwen2.5-VL + standalone MapAnything with direct geometry injection to DiT action head.

    Semantic path: Qwen2.5-VL encodes images + language, multi-layer hidden states feed DiT.
    Geometry path: MapAnything extracts 3D features, compressed to K tokens, injected into DiT cross-attention.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # --- Semantic path: Qwen2.5-VL ---
        # _QWen_VL_Interface reads from config.framework.qwenvl, so mirror the base_vlm path
        qm_cfg = self.config.framework.qwen_mapanything
        if not hasattr(self.config.framework, "qwenvl") or self.config.framework.qwenvl is None:
            from omegaconf import OmegaConf
            self.config.framework.qwenvl = OmegaConf.create(
                {"base_vlm": qm_cfg.base_vlm}
            )
        elif not hasattr(self.config.framework.qwenvl, "base_vlm"):
            self.config.framework.qwenvl.base_vlm = qm_cfg.base_vlm

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        num_vl_layers = getattr(
            self.qwen_vl_interface.model.config, "num_hidden_layers", 36
        )

        # Write VL dimensions to config for action head to read
        qm_cfg = self.config.framework.qwen_mapanything
        qm_cfg.vl_hidden_dim = llm_hidden_size
        qm_cfg.num_vl_layers = num_vl_layers
        # Also mirror to qwenvl so action head finds them regardless of lookup order
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # --- Geometric path: standalone MapAnything ---
        mapanything_model_path = getattr(qm_cfg, "mapanything_model_path", None)
        if mapanything_model_path is None:
            raise ValueError(
                "framework.qwen_mapanything.mapanything_model_path must be set"
            )

        class _MACfg:
            def __init__(self, path):
                self.mapanything_model_name_or_path = path

        self.mapanything_encoder = MapAnythingWrapper(_MACfg(mapanything_model_path))
        geom_dim = self.mapanything_encoder.config.hidden_size
        logger.info(
            f"[QwenMapAnythingPI] Loaded MapAnything standalone: "
            f"geom_dim={geom_dim}, model_path={mapanything_model_path}"
        )

        # --- Action head ---
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(
            config=self.config
        )

        # --- Geometry compression ---
        self.geom_inject_to_dit = bool(
            getattr(self.config.framework.action_model, "geom_inject_to_dit", False)
        )
        self.geom_compressor = None
        if self.geom_inject_to_dit:
            self.geom_compressor = build_geom_compressor(
                geom_dim=geom_dim,
                hidden_dim=llm_hidden_size,
                config=self.config,
            )
            compression_type = getattr(
                self.config.framework.action_model, "geom_compression_type", "pool"
            )
            num_k = int(
                getattr(self.config.framework.action_model, "geom_dit_tokens", 16)
            )
            logger.info(
                f"[geom_inject] Enabled: type={compression_type}, K={num_k}, "
                f"geom_dim={geom_dim}, hidden_dim={llm_hidden_size}"
            )

        # --- Geometry feature cache (MapAnything is frozen → same input = same output) ---
        self._geom_cache = {}
        self._geom_cache_hits = 0
        self._geom_cache_misses = 0

        # --- Config caching ---
        self.future_action_window_size = (
            config.framework.action_model.future_action_window_size
        )
        self.past_action_window_size = (
            config.framework.action_model.past_action_window_size
        )
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Image size for MapAnything preprocessing
        self.mapanything_image_size = tuple(
            getattr(qm_cfg, "mapanything_image_size", [224, 224])
        )

    def _get_repeated_diffusion_steps(self):
        repeated_diffusion_steps = 4
        if self.config is not None:
            try:
                repeated_diffusion_steps = int(
                    getattr(
                        self.config.framework.action_model,
                        "repeated_diffusion_steps",
                        repeated_diffusion_steps,
                    )
                )
            except Exception:
                pass
            try:
                if hasattr(self.config, "trainer") and hasattr(
                    self.config.trainer, "repeated_diffusion_steps"
                ):
                    repeated_diffusion_steps = int(
                        self.config.trainer.repeated_diffusion_steps
                    )
            except Exception:
                pass
        return max(1, repeated_diffusion_steps)

    def _run_mapanything(self, batch_images, device):
        """Run MapAnything on batch images and return raw geometric features.

        Uses an in-memory cache keyed by image content hash. Since MapAnything
        is frozen, the same input image always produces identical features.
        After the first epoch all frames are cached and MapAnything is skipped.

        Args:
            batch_images: list of list of PIL.Image
            device: torch device
        Returns:
            raw_geom: [B, G, geom_dim] tensor, or None if geometry injection disabled
        """
        import hashlib

        if not self.geom_inject_to_dit or self.geom_compressor is None:
            return None

        # Compute hash key for each sample's images
        cache_keys = []
        for sample_images in batch_images:
            hasher = hashlib.md5()
            for img in sample_images:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.asarray(img))
                hasher.update(img.tobytes())
            cache_keys.append(hasher.hexdigest())

        # Check if ALL samples are cached
        all_cached = all(k in self._geom_cache for k in cache_keys)

        if all_cached:
            self._geom_cache_hits += len(cache_keys)
            cached = [
                self._geom_cache[k].to(device=device, dtype=torch.float32)
                for k in cache_keys
            ]
            return torch.stack(cached)

        # Cache miss — run MapAnything on full batch
        self._geom_cache_misses += len(cache_keys)
        pixel_values = _pil_images_to_mapanything_tensor(
            batch_images,
            target_size=self.mapanything_image_size,
            device=device,
        )
        # MapAnything expects float images; it handles DINOv2 normalization internally
        with torch.autocast("cuda", dtype=torch.float32):
            geom_out = self.mapanything_encoder(
                pixel_values=pixel_values, intrinsics=None
            )
        raw_geom = geom_out.last_hidden_state  # [B, G, geom_dim]

        # Store each sample in cache (CPU, bf16 to save memory)
        for i, key in enumerate(cache_keys):
            if key not in self._geom_cache:
                self._geom_cache[key] = raw_geom[i].detach().cpu().to(torch.bfloat16)

        # Log cache stats periodically
        total = self._geom_cache_hits + self._geom_cache_misses
        if total % 1000 < len(cache_keys):
            hit_rate = self._geom_cache_hits / total * 100 if total > 0 else 0
            logger.info(
                f"[GeomCache] {len(self._geom_cache)} entries, "
                f"hit rate: {hit_rate:.1f}% ({self._geom_cache_hits}/{total})"
            )

        return raw_geom

    def _prepare_geom_for_dit(self, raw_geom, repeat_n=1):
        """Compress geometry and prepare for DiT injection.

        Returns:
            geom_tokens_for_dit: tensor or None
            geom_compressor_for_dit: PerLayerProjectionCompressor or None
        """
        if raw_geom is None:
            return None, None

        # Cast to compressor dtype (MapAnything outputs fp32, but DeepSpeed
        # mixed precision may convert compressor weights to bf16)
        target_dtype = next(self.geom_compressor.parameters()).dtype
        raw_geom = raw_geom.to(dtype=target_dtype)

        if isinstance(self.geom_compressor, PerLayerProjectionCompressor):
            geom_tokens_for_dit = raw_geom
            if repeat_n > 1:
                geom_tokens_for_dit = geom_tokens_for_dit.repeat(repeat_n, 1, 1)
            return geom_tokens_for_dit, self.geom_compressor
        else:
            geom_tokens_for_dit = self.geom_compressor(raw_geom)
            if repeat_n > 1:
                geom_tokens_for_dit = geom_tokens_for_dit.repeat(repeat_n, 1, 1)
            return geom_tokens_for_dit, None

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        # --- Semantic path: Qwen2.5-VL ---
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        # --- Geometric path: MapAnything ---
        raw_geom = self._run_mapanything(batch_images, device=base_hidden.device)

        # --- Action head ---
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = self._get_repeated_diffusion_steps()
            actions_target_repeated = actions_target.repeat(
                repeated_diffusion_steps, 1, 1
            )
            vl_embs_list_repeated = [
                h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list
            ]

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            geom_tokens_for_dit, geom_compressor_for_dit = self._prepare_geom_for_dit(
                raw_geom, repeat_n=repeated_diffusion_steps
            )

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                geom_tokens=geom_tokens_for_dit,
                geom_compressor=geom_compressor_for_dit,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        from deployment.model_server.tools.image_tools import to_pil_preserve

        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "image_size", None
        )
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # --- Semantic path ---
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        # --- Geometric path ---
        raw_geom = self._run_mapanything(batch_images, device=base_hidden.device)
        geom_tokens_for_dit, geom_compressor_for_dit = self._prepare_geom_for_dit(
            raw_geom
        )

        state_tensor = (
            torch.from_numpy(np.array(state)).to(
                base_hidden.device, dtype=base_hidden.dtype
            )
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state_tensor,
                geom_tokens=geom_tokens_for_dit,
                geom_compressor=geom_compressor_for_dit,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_train_libero_qwen_mapanything_geom_inject.yaml",
    )
    parser.add_argument(
        "--geom_compression_type",
        type=str,
        default="pool",
        choices=["pool", "per_layer", "qformer"],
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.geom_compression_type = args.geom_compression_type

    print(f"=== Testing QwenMapAnythingPI (geom_compression_type={args.geom_compression_type}) ===")
    print("Step 1: Building model...")
    model = QwenMapAnything_PI(cfg)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fake batch
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image, image],  # two views
        "lang": "Pick up the red cup and place it on the plate.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32),
    }
    batch = [sample, sample]  # batch size 2

    print("\nStep 2: Forward pass (training)...")
    output = model(batch)
    action_loss = output["action_loss"]
    print(f"  Action Loss: {action_loss.item()}")

    print("\nStep 3: Backward pass (gradient check)...")
    action_loss.backward()
    if model.geom_compressor is not None:
        has_grad = any(p.grad is not None for p in model.geom_compressor.parameters())
        print(f"  Geom compressor gradients: {has_grad}")
    print("  Backward pass OK")

    print("\nStep 4: Predict action (inference)...")
    pred = model.predict_action([sample])
    actions = pred["normalized_actions"]
    print(f"  Predicted actions shape: {actions.shape}")
    print(f"  Actions sample: {actions[0, 0, :]}")

    print(f"\n=== ALL TESTS PASSED (geom_compression_type={args.geom_compression_type}) ===")
