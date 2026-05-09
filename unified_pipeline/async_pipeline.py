"""
async_pipeline.py — Async Queue-Based Unified Pipeline Orchestrator
====================================================================
Connects all three stages via asyncio queues:

    [Moshi Token Streamer]
          ↓  token_queue  (TokenFrame @ 12.5 Hz)
    [Streaming Bridge]
          ↓  feature_queue (AudioFeatureChunk @ 25 Hz)
    [AudioFeatureBuffer]
          ↓  (accumulated features)
    [StreamingAvatarGenerator]
          ↓  frame_queue  (VideoFrame @ 25 fps)
    [Output Consumer / WebSocket]

All three stages run as concurrent asyncio tasks on the same event loop.
Thread executors are used for heavy GPU computation to avoid blocking.

Key latency optimizations:
  - Warmup: Moshi runs N frames ahead to fill the bridge pipeline.
  - Non-blocking: All queues have bounded depth; producers back-pressure.
  - CUDA streams: Bridge and AvatarForcing can run concurrently on GPU.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Optional, AsyncGenerator

import torch
import numpy as np

from .moshi_token_streamer import MoshiTokenStreamer, TokenFrame
from .bridge_streamer import StreamingBridge, AudioFeatureBuffer, AudioFeatureChunk, BridgeAudioProjection
from .avatarforcing_streamer import StreamingAvatarGenerator, VideoFrame, load_reference_image
from .config import PipelineConfig, DEFAULT_AVATAR_PROMPT

logger = logging.getLogger(__name__)

# Sentinel value for end-of-stream signaling
_EOS = object()


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedPipeline
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedPipeline:
    """
    Orchestrates the full Moshi → Bridge → AvatarForcing streaming pipeline.

    Usage:
        pipeline = UnifiedPipeline(cfg, models)
        pipeline.setup_reference_image("path/to/face.jpg")

        async for frame in pipeline.run(user_audio_pcm):
            # frame: VideoFrame with .pixels (H, W, 3) uint8
            display(frame)

    Or for continuous real-time mode:
        pipeline.start_continuous()
        await pipeline.push_audio(chunk)
        async for frame in pipeline.frame_stream():
            ...
    """

    def __init__(self, cfg: PipelineConfig, models: dict):
        self.cfg = cfg
        self.device = torch.device(cfg.avatar.device)
        self.dtype = cfg.avatar.dtype

        # ── Unpack models ──────────────────────────────────────────────────
        self.mimi = models["mimi"]
        self.lm_gen = models["lm_gen"]
        self.bridge_model = models["bridge"]
        self.projection = models["projection"]
        self.pipeline = models["pipeline"]

        # ── Build stage objects ────────────────────────────────────────────

        # Stage 1: Moshi token streamer
        self.token_streamer = MoshiTokenStreamer(
            mimi=self.mimi,
            lm_gen=self.lm_gen,
            device=self.device,
            queue_maxsize=cfg.token_queue_maxsize,
        )

        # Stage 2: Bridge
        self.bridge_streamer = StreamingBridge(
            bridge_model=self.bridge_model,
            projection=self.projection,
            device=self.device,
            dtype=self.dtype,
            chunk_tokens=cfg.bridge.chunk_tokens,
            upsample_factor=cfg.bridge.upsample_factor,
            use_kv_cache=cfg.bridge.use_kv_cache,
        )

        # Audio feature buffer (grows as bridge produces features)
        audio_dim = cfg.bridge.target_audio_dim if cfg.bridge.use_projection else cfg.bridge.output_dim
        self.feature_buffer = AudioFeatureBuffer(
            audio_dim=audio_dim,
            device=self.device,
        )

        # Stage 3: AvatarForcing
        self.avatar_generator = StreamingAvatarGenerator(
            pipeline=self.pipeline,
            device=self.device,
            dtype=self.dtype,
            num_frame_per_block=self.pipeline.num_frame_per_block,
            prompt=cfg.avatar.prompt,
            audio_dim=audio_dim,
            context_noise=cfg.avatar.config.context_noise if hasattr(cfg.avatar, 'config') else 0,
            seed=cfg.avatar.seed,
        )

        # Reference image state
        self._image_ready = False

        logger.info(
            "UnifiedPipeline ready | audio_dim=%d | chunk_tokens=%d | block_size=%d",
            audio_dim, cfg.bridge.chunk_tokens, self.pipeline.num_frame_per_block,
        )

    # ── Reference image setup ─────────────────────────────────────────────────

    def setup_reference_image(self, image_path: str) -> None:
        """
        Load and encode the reference portrait image.
        Must be called before run().
        """
        image_tensor = load_reference_image(image_path, self.device, self.dtype)
        self.avatar_generator.setup_reference_image(image_tensor)
        self._image_ready = True
        logger.info("Reference image ready: %s", image_path)

    # ── One-shot mode: process a pre-recorded audio file ─────────────────────

    async def run(
        self,
        user_audio_pcm: torch.Tensor,
        max_blocks: int = 64,
        on_frame: Optional[Callable[[VideoFrame], None]] = None,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Full pipeline run for a single audio input.

        Args:
            user_audio_pcm: [1, 1, N_samples] float32 at 24kHz
            max_blocks: safety limit on generated blocks
            on_frame: optional callback called for each VideoFrame

        Yields:
            VideoFrame objects as they are generated
        """
        if not self._image_ready:
            raise RuntimeError("Call setup_reference_image() before run()")

        # Per-run queues
        token_queue: asyncio.Queue = asyncio.Queue(maxsize=self.cfg.token_queue_maxsize)
        feature_queue: asyncio.Queue = asyncio.Queue(maxsize=self.cfg.feature_queue_maxsize)
        frame_queue: asyncio.Queue = asyncio.Queue(maxsize=self.cfg.frame_queue_maxsize)

        # Reset all streaming states
        self.bridge_streamer.reset()
        self.feature_buffer.reset()
        t_run_start = time.monotonic()

        logger.info(
            "Pipeline run started | audio_samples=%d | max_blocks=%d",
            user_audio_pcm.shape[-1], max_blocks,
        )

        # ── Task 1: Moshi token streaming ─────────────────────────────────────
        async def moshi_task():
            """Stream Moshi tokens into token_queue."""
            async for frame in self.token_streamer.stream(user_audio_pcm):
                await token_queue.put(frame)
                logger.debug("Token frame %d queued", frame.frame_idx)
            await token_queue.put(_EOS)
            logger.info("Moshi task complete: %d frames", self.token_streamer._frame_idx)

        # ── Task 2: Bridge processing ─────────────────────────────────────────
        async def bridge_task():
            """Drain token_queue → bridge → feature_queue."""
            self.bridge_streamer.reset()
            while True:
                token_frame = await token_queue.get()
                if token_frame is _EOS:
                    # Flush remaining tokens
                    chunk = await self.bridge_streamer.flush()
                    if chunk is not None:
                        await feature_queue.put(chunk)
                    await feature_queue.put(_EOS)
                    token_queue.task_done()
                    break

                chunk = await self.bridge_streamer.push_token_frame(token_frame)
                token_queue.task_done()

                if chunk is not None:
                    logger.debug(
                        "Bridge chunk: %d→%d tokens → %d audio feats",
                        chunk.token_start, chunk.token_end, chunk.features.shape[1],
                    )
                    await feature_queue.put(chunk)

            logger.info("Bridge task complete")

        # ── Task 3: Feature buffer filling ───────────────────────────────────
        # (Runs as part of avatar generation — feature_queue drained inside
        #  generate_blocks() coroutine)

        # ── Task 4: AvatarForcing streaming generation ────────────────────────
        async def avatar_task():
            """Consume features → generate video frames."""
            await self.avatar_generator.generate_blocks(
                feature_buffer=self.feature_buffer,
                feature_queue=feature_queue,
                frame_queue=frame_queue,
                total_blocks=max_blocks,
                sentinel=_EOS,
            )
            logger.info("Avatar task complete")

        # ── Run all tasks concurrently ────────────────────────────────────────
        t1 = asyncio.create_task(moshi_task(), name="moshi")
        t2 = asyncio.create_task(bridge_task(), name="bridge")
        t3 = asyncio.create_task(avatar_task(), name="avatar")

        try:
            # Yield frames as they arrive
            while True:
                frame = await frame_queue.get()
                if frame is _EOS:
                    frame_queue.task_done()
                    break

                frame_queue.task_done()
                latency_ms = (frame.timestamp - t_run_start) * 1000
                logger.debug("Frame %d ready | latency=%.1fms", frame.frame_idx, latency_ms)

                if on_frame is not None:
                    on_frame(frame)
                yield frame

        finally:
            # Ensure all tasks complete cleanly
            for task in [t1, t2, t3]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        total_time = time.monotonic() - t_run_start
        logger.info(
            "Pipeline complete | frames=%d | total_time=%.2fs | fps=%.1f",
            self.avatar_generator._generated_frames,
            total_time,
            self.avatar_generator._generated_frames / max(total_time, 0.001),
        )

    # ── Batch run: saves full video ───────────────────────────────────────────

    async def run_to_video(
        self,
        user_audio_pcm: torch.Tensor,
        output_path: str,
        fps: int = 25,
        max_blocks: int = 64,
    ) -> str:
        """
        Run full pipeline and save output as MP4 video.

        Returns: path to saved video file.
        """
        frames = []
        async for frame in self.run(user_audio_pcm, max_blocks=max_blocks):
            frames.append(frame.pixels)   # (H, W, 3) uint8

        if not frames:
            raise RuntimeError("No frames generated")

        try:
            import imageio
            import numpy as np
        except ImportError:
            logger.warning("imageio not found. Installing imageio and imageio-ffmpeg automatically...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg"])
            import imageio
            import numpy as np

        video_np = torch.stack(frames, dim=0).numpy()   # (T, H, W, 3) uint8
        
        writer = imageio.get_writer(output_path, fps=fps, macro_block_size=8)
        for frame_np in video_np:
            writer.append_data(frame_np)
        writer.close()
        
        logger.info("Video saved: %s | %d frames @ %d fps", output_path, len(frames), fps)
        return output_path

    # ── Latency benchmark ─────────────────────────────────────────────────────

    async def benchmark_latency(self, audio_seconds: float = 5.0) -> dict:
        """
        Run pipeline on synthetic audio and measure per-stage latency.
        Returns dict with timing breakdowns.
        """
        sample_rate = self.mimi.sample_rate
        n_samples = int(audio_seconds * sample_rate)
        synthetic_audio = torch.randn(1, 1, n_samples, device=self.device) * 0.1

        timings = {"first_token_ms": None, "first_feature_ms": None, "first_frame_ms": None}
        t0 = time.monotonic()

        frame_count = 0
        async for frame in self.run(synthetic_audio, max_blocks=10):
            if frame_count == 0:
                timings["first_frame_ms"] = (time.monotonic() - t0) * 1000
            frame_count += 1
            if frame_count >= 5:
                break

        timings["frames_generated"] = frame_count
        return timings


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket-ready streaming pipeline
# ─────────────────────────────────────────────────────────────────────────────

class StreamingPipelineServer:
    """
    WebSocket-ready wrapper around UnifiedPipeline.
    Handles:
      - Binary audio chunk ingestion from WebSocket
      - JPEG-encoded video frame streaming to WebSocket
      - Client connection lifecycle
    """

    def __init__(self, pipeline: UnifiedPipeline):
        self.pipeline = pipeline
        self._audio_buffer = []
        self._sample_rate = 24000

    async def handle_client(self, websocket, path=None):
        """
        WebSocket handler coroutine.
        Protocol:
          Client → Server: raw audio bytes (float32, mono, 24kHz)
          Server → Client: JPEG-encoded frame bytes prefixed with 4-byte length
        """
        import io
        from PIL import Image as PILImage

        logger.info("WebSocket client connected")
        audio_chunks = []

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Accumulate audio chunks
                    chunk = torch.frombuffer(
                        bytearray(message), dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)    # [1, 1, N]
                    audio_chunks.append(chunk)

                elif isinstance(message, str) and message == "generate":
                    # Client signals end of audio upload — start generation
                    if not audio_chunks:
                        await websocket.send(b"ERROR:no_audio")
                        continue

                    full_audio = torch.cat(audio_chunks, dim=-1).to(self.pipeline.device)
                    audio_chunks.clear()

                    async for frame in self.pipeline.run(full_audio):
                        # Encode frame as JPEG
                        pil_img = PILImage.fromarray(frame.pixels.numpy())
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=85)
                        frame_bytes = buf.getvalue()

                        # Prefix with 4-byte length
                        import struct
                        header = struct.pack(">I", len(frame_bytes))
                        await websocket.send(header + frame_bytes)

                    await websocket.send(b"EOS")

        except Exception as e:
            logger.error("WebSocket error: %s", e, exc_info=True)
        finally:
            logger.info("WebSocket client disconnected")

    async def serve(self, host: str = "0.0.0.0", port: int = 8765):
        """Start WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError("pip install websockets")

        logger.info("WebSocket server starting on %s:%d", host, port)
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()   # run forever
