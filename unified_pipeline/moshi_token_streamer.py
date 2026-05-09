"""
moshi_token_streamer.py — Moshi Discrete Token Streaming Extractor
====================================================================
Hooks into Moshi's LMGen to intercept audio tokens BEFORE waveform decoding.
Streams (B=1, 8) token frames at 12.5 Hz via an asyncio Queue.

Design:
  - on_audio_hook callback fires every time LMGen generates one audio frame
  - Tokens are placed on a thread-safe asyncio Queue
  - Caller reads from queue without blocking the inference thread
  - Mimi encoder is used for encoding USER input audio
  - Moshi generates ASSISTANT speech tokens continuously
"""

import asyncio
import logging
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Token frame dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenFrame:
    """One Moshi audio token frame: 8 codebook values at 12.5 Hz."""
    tokens: torch.Tensor    # shape (8,) — int64 Mimi codebook indices
    timestamp: float        # wall-clock time when emitted
    frame_idx: int          # monotonic frame counter


# ─────────────────────────────────────────────────────────────────────────────
# MoshiTokenStreamer
# ─────────────────────────────────────────────────────────────────────────────

class MoshiTokenStreamer:
    """
    Wraps Moshi's LMGen inference loop to extract discrete audio tokens
    frame-by-frame as they are generated.

    Usage (inside an asyncio context):
        streamer = MoshiTokenStreamer(mimi, lm_gen, device)
        async for frame in streamer.stream(user_audio_pcm):
            # frame.tokens: (8,) int64 Mimi codebook indices
            await bridge_queue.put(frame)

    The user_audio_pcm tensor is shape [1, 1, N_samples] at 24kHz.
    """

    def __init__(
        self,
        mimi,
        lm_gen,
        device: torch.device,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        queue_maxsize: int = 256,
    ):
        self.mimi = mimi
        self.lm_gen = lm_gen
        self.device = device
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._frame_idx = 0
        self._stop_event = threading.Event()

        # frame_size: number of audio samples per Mimi frame
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)   # 24000/12.5 = 1920

    # ── Internal callback installed on LMGen ─────────────────────────────────

    def _on_audio_hook(self, audio_tokens: torch.Tensor) -> None:
        """
        Called by LMGen every time it generates one audio frame.
        audio_tokens: shape (B, dep_q) — batch × num_codebooks (B=1, dep_q=8)

        This runs IN THE INFERENCE THREAD (not asyncio thread).
        We post to the asyncio queue in a thread-safe way.
        """
        if self._stop_event.is_set():
            return

        # Take first batch item; detach immediately to free computation graph
        tokens_1d = audio_tokens[0].detach().cpu()   # shape (8,)

        frame = TokenFrame(
            tokens=tokens_1d,
            timestamp=time.monotonic(),
            frame_idx=self._frame_idx,
        )
        self._frame_idx += 1

        if self._loop is not None and self._loop.is_running():
            # Thread-safe put into asyncio queue
            asyncio.run_coroutine_threadsafe(
                self._queue.put(frame), self._loop
            )
        else:
            # Fallback: best-effort synchronous put (used in non-async contexts)
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("TokenStream queue full — dropping frame %d", frame.frame_idx)

    # ── Main async generator ──────────────────────────────────────────────────

    async def stream(
        self,
        in_pcms: torch.Tensor,
        sentinel_timeout: float = 2.0,
    ) -> AsyncGenerator[TokenFrame, None]:
        """
        Async generator that yields TokenFrame objects as Moshi generates them.

        Args:
            in_pcms: User speech audio tensor [B, 1, N_samples] at 24kHz
            sentinel_timeout: seconds to wait for the final token after inference ends

        Yields:
            TokenFrame — one per Mimi audio frame (12.5 Hz)
        """
        self._frame_idx = 0
        self._stop_event.clear()
        loop = asyncio.get_event_loop()
        self._loop = loop

        # Install hook on LMGen
        self.lm_gen.on_audio_hook = self._on_audio_hook

        # Split into Mimi frames
        chunks = deque(
            chunk
            for chunk in in_pcms.split(self.frame_size, dim=2)
            if chunk.shape[-1] == self.frame_size
        )

        try:
            with torch.no_grad():
                first_frame = True
                while chunks and not self._stop_event.is_set():
                    chunk = chunks.popleft()
                    codes = self.mimi.encode(chunk)   # [B, n_q, 1]

                    if first_frame:
                        tokens = self.lm_gen.step(codes)
                        if max(self.lm_gen.lm_model.delays) > 0:
                            assert tokens is None
                        first_frame = False

                    tokens = self.lm_gen.step(codes)
                    
                    # Yield any generated tokens from the queue
                    while not self._queue.empty():
                        frame = self._queue.get_nowait()
                        yield frame
                        self._queue.task_done()
                    
                    # Yield control to the event loop so other tasks can run
                    await asyncio.sleep(0)

            # Drain any remaining frames
            deadline = loop.time() + sentinel_timeout
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=min(remaining, 0.05))
                    yield frame
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    break

        except Exception as exc:
            logger.error("Moshi inference error: %s", exc, exc_info=True)
            raise
        finally:
            self.lm_gen.on_audio_hook = None
            logger.debug("Token stream complete: %d frames emitted", self._frame_idx)

    def stop(self) -> None:
        """Signal the inference thread to stop early."""
        self._stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# ContinuousTokenStreamer — for real-time microphone input (server mode)
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousTokenStreamer:
    """
    Streaming token extractor for CONTINUOUS real-time conversation.
    Accepts audio chunks pushed externally (e.g., from WebSocket client)
    and continuously runs Moshi inference, yielding token frames.

    The Moshi streaming state (KV cache, codebook delays) persists across
    audio chunks — this is the key to real-time low-latency operation.

    Usage:
        streamer = ContinuousTokenStreamer(mimi, lm_gen, device)
        streamer.start()                           # begin inference loop

        # Push user audio chunks as they arrive (from WebSocket):
        await streamer.push_audio(pcm_chunk)       # [1, 1, 1920] at 24kHz

        # Consume token frames from another coroutine:
        async for frame in streamer.token_frames():
            ...
    """

    def __init__(
        self,
        mimi,
        lm_gen,
        device: torch.device,
        audio_queue_maxsize: int = 64,
        token_queue_maxsize: int = 256,
    ):
        self.mimi = mimi
        self.lm_gen = lm_gen
        self.device = device
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)

        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=audio_queue_maxsize)
        self._token_queue: asyncio.Queue = asyncio.Queue(maxsize=token_queue_maxsize)
        self._frame_idx = 0
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def push_audio(self, pcm_chunk: torch.Tensor) -> None:
        """
        Push one audio frame to the inference queue.
        pcm_chunk: [1, 1, frame_size] float32 at 24kHz.
        Non-blocking: drops frame if queue is full.
        """
        try:
            self._audio_queue.put_nowait(pcm_chunk)
        except asyncio.QueueFull:
            logger.warning("Audio queue full — dropping audio chunk")

    def _on_audio_hook(self, audio_tokens: torch.Tensor) -> None:
        """Called by LMGen for every generated audio frame (from inference thread)."""
        tokens_1d = audio_tokens[0].detach().cpu()
        frame = TokenFrame(
            tokens=tokens_1d,
            timestamp=time.monotonic(),
            frame_idx=self._frame_idx,
        )
        self._frame_idx += 1

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._token_queue.put(frame), self._loop
            )

    def start(self) -> None:
        """Start the background inference thread."""
        self._loop = asyncio.get_event_loop()
        self.lm_gen.on_audio_hook = self._on_audio_hook
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="moshi-inference",
        )
        self._inference_thread.start()
        logger.info("ContinuousTokenStreamer started")

    def stop(self) -> None:
        """Gracefully stop the inference thread."""
        self._running = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
        self.lm_gen.on_audio_hook = None
        logger.info("ContinuousTokenStreamer stopped")

    def _inference_loop(self) -> None:
        """
        Background thread: drains audio queue and runs Moshi step-by-step.
        The Moshi streaming context must already be active (lm_gen.streaming_forever).
        """
        silence = torch.zeros(1, 1, self.frame_size, device=self.device)

        with torch.no_grad():
            first_frame = True
            while self._running:
                # Block waiting for audio (max 100ms to check stop flag)
                try:
                    loop = asyncio.new_event_loop()
                    future = asyncio.run_coroutine_threadsafe(
                        self._audio_queue.get(), self._loop
                    )
                    chunk = future.result(timeout=0.1)
                except Exception:
                    # No audio — feed silence so Moshi can keep responding
                    chunk = silence.clone()

                chunk = chunk.to(self.device)
                codes = self.mimi.encode(chunk)

                if first_frame:
                    self.lm_gen.step(codes)
                    if max(self.lm_gen.lm_model.delays) > 0:
                        first_frame = False
                        continue
                    first_frame = False

                self.lm_gen.step(codes)

    async def token_frames(self) -> AsyncGenerator[TokenFrame, None]:
        """Async generator that yields TokenFrame objects from the token queue."""
        while self._running:
            try:
                frame = await asyncio.wait_for(self._token_queue.get(), timeout=0.1)
                yield frame
                self._token_queue.task_done()
            except asyncio.TimeoutError:
                continue
