"""
bridge_streamer.py — Streaming Bridge Inference
=================================================
Converts Moshi discrete token frames (12.5 Hz, 8 codebooks)
→ continuous wav2vec2-like audio features (25 Hz, 768-dim → projected to 9984-dim).

Key design decisions:
  1. KV-cache within the bridge transformer for truly causal streaming.
  2. Chunk accumulation: wait for N token frames before processing
     (reduces per-step overhead while keeping latency manageable).
  3. BridgeAudioProjection: projects 768 → 9984 to be drop-in compatible
     with AvatarForcing's concatenated wav2vec feature format.
  4. Thread-safe asyncio queue interface.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Feature Frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioFeatureChunk:
    """Bridge output: a chunk of continuous wav2vec-like audio features."""
    features: torch.Tensor     # shape (1, T_out, audio_dim) — already projected
    token_start: int           # first Moshi token frame index in this chunk
    token_end: int             # last+1 Moshi token frame index
    timestamp: float


# ─────────────────────────────────────────────────────────────────────────────
# BridgeAudioProjection
# ─────────────────────────────────────────────────────────────────────────────

class BridgeAudioProjection(nn.Module):
    """
    Lightweight linear projection: bridge_out_dim (768) → target_audio_dim.

    AvatarForcing's Wav2VecModel.forward() is called with output_hidden_states=True,
    then the dataset concatenates last_hidden_state + all 12 encoder hidden states:
        cat([last_hidden_state, *hidden_states], dim=-1)  → (B, T, 768 × 14) = (B, T, 10752)

    We replicate the same final tensor shape so that conditional_dict["audio_emb"]
    is compatible with CausalWanModel's internal projection layers — no WanModel
    weights need to be changed.

    Architecture:
        Linear(768, 10752)  +  LayerNorm(10752)
    Small enough (~7.6M params) to load alongside the bridge checkpoint.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 10752):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 768) → (B, T, 9984)"""
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# StreamingBridge
# ─────────────────────────────────────────────────────────────────────────────

class StreamingBridge:
    """
    Stateful streaming bridge: consumes TokenFrame objects and outputs
    AudioFeatureChunk objects suitable for direct injection into AvatarForcing.

    State management:
        - KV cache: persists the bridge transformer's key-value pairs
          across chunks → true causal streaming without context loss
        - token_buffer: accumulates token frames until chunk_size reached
        - frame_counter: tracks global frame position for alignment

    Thread safety:
        All public methods are coroutines and safe to call from asyncio.
        The heavy forward pass runs in a thread executor.
    """

    def __init__(
        self,
        bridge_model: nn.Module,
        projection: BridgeAudioProjection,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        chunk_tokens: int = 4,
        upsample_factor: int = 2,
        use_kv_cache: bool = True,
    ):
        self.bridge = bridge_model
        self.projection = projection
        self.device = device
        self.dtype = dtype
        self.chunk_tokens = chunk_tokens          # accumulate N tokens per forward pass
        self.upsample_factor = upsample_factor    # bridge upsampling ratio
        self.use_kv_cache = use_kv_cache

        # Streaming state
        self._token_buffer: List[torch.Tensor] = []   # list of (8,) int64 tensors
        self._kv_cache = None                          # bridge KV cache (lazily init)
        self._global_token_idx: int = 0               # total tokens consumed
        self._global_feature_idx: int = 0             # total feature frames produced

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset streaming state for a new conversation."""
        self._token_buffer.clear()
        self._kv_cache = None
        self._global_token_idx = 0
        self._global_feature_idx = 0
        # Reset bridge transformer KV cache
        if hasattr(self.bridge, 'transformer') and hasattr(self.bridge.transformer, 'layers'):
            for layer in self.bridge.transformer.layers:
                if hasattr(layer, 'attn'):
                    layer.attn.reset_cache()
        logger.debug("StreamingBridge state reset")

    async def push_token_frame(
        self,
        token_frame,                           # TokenFrame from moshi_token_streamer
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Optional[AudioFeatureChunk]:
        """
        Accept one Moshi token frame.
        Returns an AudioFeatureChunk when the chunk buffer is full, else None.

        This is the primary streaming interface: call once per TokenFrame.
        """
        self._token_buffer.append(token_frame.tokens)    # (8,) int64

        if len(self._token_buffer) < self.chunk_tokens:
            return None

        # Enough tokens: run bridge forward pass sequentially
        chunk = self._forward_chunk()
        # Yield control
        await asyncio.sleep(0)
        return chunk

    async def flush(self) -> Optional[AudioFeatureChunk]:
        """
        Process any remaining tokens in the buffer (< chunk_tokens).
        Call at end of Moshi stream to drain the pipeline.
        """
        if not self._token_buffer:
            return None
        chunk = self._forward_chunk()
        await asyncio.sleep(0)
        return chunk

    # ── Internal forward pass (runs in thread pool) ───────────────────────────

    def _forward_chunk(self) -> AudioFeatureChunk:
        """
        Blocking forward pass. Called from thread executor.
        Pops tokens from buffer, runs bridge, returns features.
        """
        tokens_list = self._token_buffer[:]
        self._token_buffer.clear()

        T = len(tokens_list)
        t_start = self._global_token_idx
        t_end = t_start + T
        self._global_token_idx = t_end

        # Stack to (1, T, 8) — batch=1, T frames, 8 codebooks
        tokens_tensor = torch.stack(tokens_list, dim=0).unsqueeze(0)    # (1, T, 8)
        tokens_tensor = tokens_tensor.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # Bridge forward: (1, T, 8) → (1, T*upsample_factor, 768), present_kvs
                features, self._kv_cache = self.bridge(
                    tokens_tensor,
                    use_cache=self.use_kv_cache,
                    past_kvs=self._kv_cache,
                )
                # features: (1, T*upsample_factor, 768)

                # Project to AvatarForcing-compatible dim: (1, T_out, 10752)
                features = self.projection(features)
        
        # Track feature frame count
        T_out = features.shape[1]
        f_start = self._global_feature_idx
        self._global_feature_idx += T_out

        return AudioFeatureChunk(
            features=features.detach().float(),    # keep as float32 for stability
            token_start=t_start,
            token_end=t_end,
            timestamp=time.monotonic(),
        )

    # ── Async generator interface ─────────────────────────────────────────────

    async def process_stream(
        self,
        token_queue: asyncio.Queue,
        feature_queue: asyncio.Queue,
        sentinel=None,
    ) -> None:
        """
        Full pipeline coroutine: drains token_queue, processes via bridge,
        puts AudioFeatureChunk objects onto feature_queue.

        sentinel: value that signals end of stream (default None).
        """
        self.reset()
        t0 = time.monotonic()

        while True:
            token_frame = await token_queue.get()

            if token_frame is sentinel:
                # End-of-stream: flush remaining tokens
                chunk = await self.flush()
                if chunk is not None:
                    await feature_queue.put(chunk)
                await feature_queue.put(sentinel)   # propagate EOS
                token_queue.task_done()
                break

            chunk = await self.push_token_frame(token_frame)
            token_queue.task_done()

            if chunk is not None:
                latency_ms = (chunk.timestamp - t0) * 1000
                logger.debug(
                    "Bridge chunk: tokens [%d-%d] → %d audio features | lat=%.1fms",
                    chunk.token_start, chunk.token_end,
                    chunk.features.shape[1], latency_ms,
                )
                await feature_queue.put(chunk)
                t0 = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# AudioFeatureBuffer — temporal accumulator for AvatarForcing alignment
# ─────────────────────────────────────────────────────────────────────────────

class AudioFeatureBuffer:
    """
    Accumulates streaming bridge output chunks and provides aligned slices
    for AvatarForcing's sliding-window diffusion.

    AvatarForcing needs a contiguous (1, T_total, audio_dim) tensor covering
    the full duration. We build this incrementally and return views over it.

    The buffer grows as more audio features arrive. Once we have enough features
    to cover the next AvatarForcing block (num_frames_needed), we return that slice.
    """

    def __init__(self, audio_dim: int = 10752, device: torch.device = torch.device("cuda")):
        self.audio_dim = audio_dim
        self.device = device
        self._chunks: List[torch.Tensor] = []    # list of (1, T_i, audio_dim)
        self._total_frames: int = 0

    def reset(self) -> None:
        self._chunks.clear()
        self._total_frames = 0

    def push(self, chunk: AudioFeatureChunk) -> None:
        """Append a new feature chunk to the buffer."""
        feat = chunk.features.to(self.device)  # (1, T, audio_dim)
        self._chunks.append(feat)
        self._total_frames += feat.shape[1]

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def get_full_tensor(self) -> torch.Tensor:
        """
        Return the full concatenated feature tensor: (1, T_total, audio_dim).
        Adds a leading zero frame as expected by AvatarForcing's dataset pipeline.
        """
        if not self._chunks:
            return torch.zeros(1, 1, self.audio_dim, device=self.device)

        combined = torch.cat(self._chunks, dim=1)   # (1, T_total, audio_dim)

        # AvatarForcing's dataset adds a zero prefix frame:
        # sample['audio_emb'] = torch.cat([zeros_like(audio_emb[:1]), audio_emb], dim=0)
        prefix = torch.zeros(1, 1, self.audio_dim, device=combined.device, dtype=combined.dtype)
        return torch.cat([prefix, combined], dim=1)   # (1, T_total+1, audio_dim)

    def get_slice(self, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        Return a slice of the feature tensor: (1, end-start, audio_dim).
        Indices include the prefix zero frame.
        """
        full = self.get_full_tensor()               # (1, T_total+1, audio_dim)
        # Clamp to available frames
        end_frame = min(end_frame, full.shape[1])
        return full[:, start_frame:end_frame, :]    # (1, slice_len, audio_dim)

    def has_enough(self, frames_needed: int) -> bool:
        """True if we have at least frames_needed audio feature frames (excl. prefix)."""
        return self._total_frames >= frames_needed
