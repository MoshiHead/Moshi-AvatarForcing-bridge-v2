"""
model_loader.py — Centralized Model Loading & Initialization
=============================================================
Loads all three models (Moshi, Bridge, AvatarForcing) with:
  - Mixed precision (bfloat16)
  - VRAM-efficient loading (CPU first, then move to GPU)
  - torch.inference_mode for all forward passes
  - Proper path resolution for RunPod environments

All models share the same CUDA device unless explicitly overridden.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_vram(tag: str, device: torch.device) -> None:
    """Log current VRAM usage for debugging."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info("[VRAM] %s → allocated=%.2f GB | reserved=%.2f GB", tag, allocated, reserved)


def _add_repo_to_path(repo_path: str) -> None:
    """Add a repository to sys.path (avoids import conflicts)."""
    abs_path = str(Path(repo_path).resolve())
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
        logger.debug("Added to sys.path: %s", abs_path)


# ─────────────────────────────────────────────────────────────────────────────
# Moshi model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_moshi_models(
    moshi_repo_path: str,
    hf_repo: str = "kyutai/moshiko-pytorch-bf16",
    moshi_weight: Optional[str] = None,
    mimi_weight: Optional[str] = None,
    tokenizer: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cfg_coef: float = 1.0,
    use_sampling: bool = True,
    temp: float = 0.8,
    temp_text: float = 0.7,
    top_k: int = 250,
):
    """
    Load Moshi LM, Mimi codec, and text tokenizer.

    Returns:
        mimi: MimiModel — audio tokenizer/detokenizer
        lm_gen: LMGen — wrapped language model for streaming inference
        text_tokenizer: SentencePiece tokenizer
        checkpoint_info: CheckpointInfo
    """
    _add_repo_to_path(moshi_repo_path)

    from moshi.models import loaders, LMGen
    from moshi.run_inference import InferenceState

    torch_device = torch.device(device)

    logger.info("Loading Moshi checkpoint from: %s", hf_repo)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo,
        moshi_weight,
        mimi_weight,
        tokenizer,
    )

    logger.info("Loading Mimi audio codec…")
    mimi = checkpoint_info.get_mimi(device=torch_device)
    mimi.eval()
    _log_vram("after Mimi", torch_device)

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    logger.info("Loading Moshi language model…")
    lm = checkpoint_info.get_moshi(device=torch_device, dtype=dtype)
    lm.eval()
    _log_vram("after Moshi LM", torch_device)

    # Wrap in LMGen for streaming inference
    from moshi.run_inference import get_condition_tensors
    batch_size = 1
    condition_tensors = get_condition_tensors(
        checkpoint_info.model_type, lm, batch_size, cfg_coef
    )
    lm_gen = LMGen(
        lm,
        use_sampling=use_sampling,
        temp=temp,
        temp_text=temp_text,
        top_k=top_k,
        cfg_coef=cfg_coef,
        condition_tensors=condition_tensors,
        **checkpoint_info.lm_gen_config,
    )

    # Initialize streaming context (keeps KV cache alive across steps)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    mimi.streaming_forever(batch_size)
    lm_gen.streaming_forever(batch_size)

    logger.info(
        "Moshi loaded | frame_size=%d | sample_rate=%d | frame_rate=%.1f",
        frame_size, mimi.sample_rate, mimi.frame_rate,
    )
    return mimi, lm_gen, text_tokenizer, checkpoint_info


# ─────────────────────────────────────────────────────────────────────────────
# Bridge model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_bridge_model(
    bridge_repo_path: str,
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    wav2vec_num_layers: int = 13,
    use_projection: bool = True,
):
    """
    Load the MimiWav2Vec2Bridge model and optional BridgeAudioProjection.

    The bridge checkpoint may contain:
        {"bridge": state_dict, ...}   — from trainer.py save_checkpoint()
        or the raw state dict directly

    Returns:
        bridge: MimiWav2Vec2Bridge — frozen for inference
        projection: BridgeAudioProjection — maps 768 → 10752
        bridge_cfg: dict — parsed config
    """
    _add_repo_to_path(bridge_repo_path)

    import yaml
    from model import MimiWav2Vec2Bridge   # from bridge_module/model.py

    torch_device = torch.device(device)

    # Parse bridge config
    with open(config_path, "r") as f:
        bridge_cfg = yaml.safe_load(f)

    logger.info("Building bridge model…")
    bridge = MimiWav2Vec2Bridge(bridge_cfg)

    # Load checkpoint
    logger.info("Loading bridge checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle both trainer-format and raw state dict
    if isinstance(ckpt, dict):
        if "bridge" in ckpt:
            state_dict = ckpt["bridge"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    bridge.load_state_dict(state_dict, strict=True)
    bridge = bridge.to(device=torch_device, dtype=dtype)
    bridge.eval()
    bridge.requires_grad_(False)

    _log_vram("after Bridge", torch_device)
    logger.info("Bridge loaded | output_dim=%d", bridge_cfg["model"]["output_dim"])

    # Optional projection layer
    projection = None
    if use_projection:
        output_dim = bridge_cfg["model"]["output_dim"]        # 768
        # Force 14 layers (10752) because AvatarForcing explicitly concatenates 14 Wav2Vec states
        # This overrides any older notebook configs that might pass wav2vec_num_layers=13
        target_dim = output_dim * 14                          # 10752

        # Try to load projection weights from checkpoint
        projection = BridgeAudioProjectionShim(output_dim, target_dim)

        proj_key = "projection"
        if isinstance(ckpt, dict) and proj_key in ckpt:
            projection.load_state_dict(ckpt[proj_key])
            logger.info("Loaded bridge projection weights from checkpoint")
        else:
            logger.warning(
                "No projection weights in checkpoint — using random init. "
                "Train BridgeAudioProjection before production use."
            )

        projection = projection.to(device=torch_device, dtype=dtype)
        projection.eval()
        projection.requires_grad_(False)

    return bridge, projection, bridge_cfg


class BridgeAudioProjectionShim(nn.Module):
    """
    Inline definition so model_loader has no circular import.
    Identical to bridge_streamer.BridgeAudioProjection.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 10752):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# AvatarForcing model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_avatarforcing_pipeline(
    avatarforcing_repo_path: str,
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_ema: bool = False,
):
    """
    Load the AvatarForcing inference pipeline.

    WAV2VEC IS NOT LOADED — we bypass it entirely.
    Only the diffusion generator, VAE, and text encoder are loaded.

    Returns:
        pipeline: AvatarForcingInferencePipeline
    """
    _add_repo_to_path(avatarforcing_repo_path)

    from omegaconf import OmegaConf
    from collections import OrderedDict
    from pipeline import AvatarForcingInferencePipeline
    from utils.inject import _apply_lora

    torch_device = torch.device(device)

    logger.info("Loading AvatarForcing config: %s", config_path)
    config = OmegaConf.load(config_path)
    default_config_path = Path(avatarforcing_repo_path) / "configs" / "default_config.yaml"
    default_config = OmegaConf.load(str(default_config_path))
    config = OmegaConf.merge(default_config, config)

    logger.info("Building AvatarForcing pipeline…")
    pipeline = AvatarForcingInferencePipeline(config, device=torch_device)

    if checkpoint_path:
        logger.info("Loading AvatarForcing checkpoint: %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if use_ema:
            state_dict_to_load = state_dict["generator_ema"]
            def _remove_fsdp(sd):
                new = OrderedDict()
                for k, v in sd.items():
                    new[k.replace("_fsdp_wrapped_module.", "")] = v
                return new
            state_dict_to_load = _remove_fsdp(state_dict_to_load)
        else:
            state_dict_to_load = state_dict["generator"]

        if hasattr(config, "models") and hasattr(config.models, "lora"):
            pipeline.generator.model = _apply_lora(pipeline.generator.model, config["models"]["lora"])

        pipeline.generator.load_state_dict(state_dict_to_load)
        logger.info("AvatarForcing checkpoint loaded")

    pipeline = pipeline.to(device=torch_device, dtype=dtype)
    pipeline.eval()

    # Disable WAV2VEC — we never call it
    # The TextImageAudioPairDataset normally loads wav2vec; we skip that entirely
    # by not using the dataset class and injecting bridge features directly.
    logger.info("WAV2VEC bypassed — bridge features will be injected directly")

    _log_vram("after AvatarForcing", torch_device)
    logger.info(
        "AvatarForcing loaded | block_size=%d | denoise_steps=%s",
        config.num_frame_per_block,
        list(pipeline.denoise_steps.tolist()),
    )
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Unified model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models(cfg) -> dict:
    """
    Load all models in the correct order (Moshi, Bridge, AvatarForcing).
    Returns a dict with keys: mimi, lm_gen, text_tokenizer, bridge, projection, pipeline.

    cfg: PipelineConfig dataclass (from config.py)
    """
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    results = {}

    # ── 1. Moshi ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading Moshi…")
    moshi_root = str(Path(__file__).parent.parent / "moshi-inference")
    mimi, lm_gen, text_tokenizer, ckpt_info = load_moshi_models(
        moshi_repo_path=moshi_root,
        hf_repo=cfg.moshi.hf_repo,
        moshi_weight=cfg.moshi.moshi_weight,
        mimi_weight=cfg.moshi.mimi_weight,
        tokenizer=cfg.moshi.tokenizer,
        device=cfg.moshi.device,
        dtype=cfg.moshi.dtype,
        cfg_coef=cfg.moshi.cfg_coef,
        use_sampling=cfg.moshi.use_sampling,
        temp=cfg.moshi.temp,
        temp_text=cfg.moshi.temp_text,
        top_k=cfg.moshi.top_k,
    )
    results.update({"mimi": mimi, "lm_gen": lm_gen, "text_tokenizer": text_tokenizer})

    torch.cuda.empty_cache()
    gc.collect()

    # ── 2. Bridge ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading Bridge…")
    bridge_root = str(Path(__file__).parent.parent / "bridge_module")
    bridge, projection, bridge_cfg = load_bridge_model(
        bridge_repo_path=bridge_root,
        checkpoint_path=cfg.bridge.checkpoint_path,
        config_path=cfg.bridge.config_path,
        device=cfg.bridge.device,
        dtype=cfg.bridge.dtype,
        wav2vec_num_layers=cfg.bridge.wav2vec_num_layers,
        use_projection=cfg.bridge.use_projection,
    )
    results.update({"bridge": bridge, "projection": projection, "bridge_cfg": bridge_cfg})

    torch.cuda.empty_cache()
    gc.collect()

    # ── 3. AvatarForcing ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading AvatarForcing…")
    af_root = str(Path(__file__).parent.parent / "AvatarForcing-inference")
    pipeline = load_avatarforcing_pipeline(
        avatarforcing_repo_path=af_root,
        config_path=cfg.avatar.config_path,
        checkpoint_path=cfg.avatar.checkpoint_path,
        device=cfg.avatar.device,
        dtype=cfg.avatar.dtype,
        use_ema=cfg.avatar.use_ema,
    )
    results["pipeline"] = pipeline

    _log_vram("ALL MODELS LOADED", torch.device(cfg.avatar.device))
    logger.info("=" * 60)
    logger.info("All models loaded successfully")

    return results
