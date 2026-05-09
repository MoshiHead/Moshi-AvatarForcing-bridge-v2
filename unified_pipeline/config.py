"""
config.py — Unified Pipeline Configuration
==========================================
Central configuration for the Moshi → Bridge → AvatarForcing streaming pipeline.
All tunable parameters live here; no magic numbers anywhere else.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Default AvatarForcing prompt (fixed for all conversations)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_AVATAR_PROMPT = (
    "A long-haired young man faces the lens against a neutral backdrop, wearing a denim shirt. He speaks in a composed, even rhythm; lip shapes are clearly articulated, with occasional faint smiles and slight brow movement for nuance. Natural blinks and small head shifts keep the portrait alive while maintaining frontal alignment. The camera is locked on a mid-close shot with a subtle push-in; depth of field isolates the subject while preserving crisp eye detail. Lighting is soft and balanced, producing consistent skin tone and stable color temperature throughout."
)


@dataclass
class MoshiConfig:
    """Moshi speech LM configuration."""
    hf_repo: str = "kyutai/moshiko-pytorch-bf16"
    moshi_weight: Optional[str] = None          # override path, None = download from HF
    mimi_weight: Optional[str] = None
    tokenizer: Optional[str] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    cfg_coef: float = 1.0
    use_sampling: bool = True
    temp: float = 0.8
    temp_text: float = 0.7
    top_k: int = 250
    # Mimi codec parameters (fixed by model)
    sample_rate: int = 24000                    # Mimi sample rate
    frame_rate: float = 12.5                    # Mimi frames/sec → token rate
    num_codebooks: int = 8                      # dep_q audio codebooks
    batch_size: int = 1


@dataclass
class BridgeConfig:
    """Mimi→Wav2Vec2 bridge model configuration."""
    checkpoint_path: str = ""                   # path to bridge checkpoint .pt file
    config_path: str = ""                       # path to bridge config.yaml
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Architecture (must match config.yaml)
    num_codebooks: int = 8
    vocab_size: int = 2048
    embed_dim: int = 256
    d_model: int = 512
    output_dim: int = 768                       # matches wav2vec2-base-960h hidden size
    upsample_factor: int = 2                    # 12.5 Hz × 2 = 25 Hz output
    # Projection to match AvatarForcing wav2vec concatenated dim
    # Original pipeline: last_hidden_state + 13 transformer hidden_states = 14 × 768 = 10752
    # We project bridge (768) → target_audio_dim for drop-in replacement
    wav2vec_num_layers: int = 14                # 1 last + 13 hidden = 14 stacked
    target_audio_dim: int = 10752               # = 10752 (what AvatarForcing expects)
    use_projection: bool = True                 # True = add linear projection layer
    # Streaming inference
    chunk_tokens: int = 4                       # Moshi token frames per bridge chunk
                                                # 4 tokens @ 12.5Hz = 320ms
    use_kv_cache: bool = True                   # Causal KV cache for streaming


@dataclass
class AvatarForcingConfig:
    """AvatarForcing diffusion pipeline configuration."""
    # Model paths (AvatarForcing-inference directory)
    avatarforcing_root: str = "../AvatarForcing-inference"
    config_path: str = "../AvatarForcing-inference/configs/avatarforcing.yaml"
    checkpoint_path: str = ""                   # path to .pt checkpoint
    wan_models_dir: str = "../AvatarForcing-inference/wan_models"
    use_ema: bool = False
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Generation parameters
    num_output_frames: int = 21                 # frames per generation block
    num_frame_per_block: int = 4               # frames processed per diffusion block
    fps: int = 25                               # output video FPS
    # Image conditioning
    image_height: int = 480
    image_width: int = 832
    # Fixed prompt (never changes during conversation)
    prompt: str = DEFAULT_AVATAR_PROMPT
    # Streaming
    seed: int = 42


@dataclass
class PipelineConfig:
    """Top-level unified pipeline configuration."""
    moshi: MoshiConfig = field(default_factory=MoshiConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    avatar: AvatarForcingConfig = field(default_factory=AvatarForcingConfig)

    # ── Queue / async settings ──────────────────────────────────────────────
    token_queue_maxsize: int = 128              # Moshi→Bridge token queue depth
    feature_queue_maxsize: int = 64            # Bridge→AvatarForcing feature queue
    frame_queue_maxsize: int = 32              # AvatarForcing→Output frame queue

    # ── Latency optimizations ───────────────────────────────────────────────
    warmup_tokens: int = 8                      # token frames to accumulate before
                                                # starting avatar generation (gives
                                                # bridge a head-start)
    target_fps: int = 25                        # target output FPS
    vae_decode_every_n_frames: int = 1          # decode latent every N frames
                                                # (set >1 to batch VAE decoding)
    # ── WebSocket server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8765
    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    profile_latency: bool = False
