"""
avatarforcing_streamer.py — AvatarForcing Streaming Generator
==============================================================
Replaces wav2vec audio features with bridge model output and generates
talking-head video frames in a streaming sliding-window fashion.

Key modifications vs. original AvatarForcing:
  1. NO wav2vec encoder — bridge features injected directly.
  2. Incremental frame generation: one diffusion block at a time as features arrive.
  3. VAE decoding per-block (or batched) for real-time pixel output.
  4. Audio feature alignment: bridge output frames ↔ video frames at 25 Hz.
  5. All KV caches preserved across blocks for temporal consistency.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import math
import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Video frame dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoFrame:
    """One decoded video frame from the avatar pipeline."""
    pixels: torch.Tensor     # (H, W, 3) uint8 RGB
    frame_idx: int
    timestamp: float
    latency_ms: float        # ms from first audio token to this frame


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (mirrors inference.py ResizeKeepRatioArea16)
# ─────────────────────────────────────────────────────────────────────────────

class ResizeKeepRatioArea16:
    """Resize image to fit within area budget, keeping ratio, div-16 aligned."""
    def __init__(self, area_hw=(480, 832), div=16):
        self.A = area_hw[0] * area_hw[1]
        self.d = div

    def __call__(self, img):
        w, h = img.size
        s = min(1.0, math.sqrt(self.A / (h * w)))
        nh = max(self.d, int(h * s) // self.d * self.d)
        nw = max(self.d, int(w * s) // self.d * self.d)
        return TF.resize(img, (nh, nw), interpolation=InterpolationMode.BILINEAR, antialias=True)


def load_reference_image(image_path: str, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess reference image for AvatarForcing I2V conditioning.

    Returns:
        image_tensor: (1, 1, 3, H, W) bfloat16 — raw pixel tensor
        initial_latent: (1, 1, 16, H//8, W//8) bfloat16 — VAE-encoded latent
    """
    transform = transforms.Compose([
        ResizeKeepRatioArea16((480, 832), 16),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img)                                   # (3, H, W)
    image_tensor = img_t.unsqueeze(0).unsqueeze(2)          # (1, 3, 1, H, W)
    # Rearrange to [B, C, T, H, W] which is what encode_to_latent expects
    # Actually encode_to_latent expects [B, C, T, H, W]
    # But looking at inference.py:
    #   image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2) -> [1, 3, 1, H, W]
    #   initial_latent = pipeline.vae.encode_to_latent(image)
    image_tensor_for_vae = image_tensor.to(device=device, dtype=dtype)
    logger.info("Reference image loaded: %s → tensor %s", image_path, list(image_tensor.shape))
    return image_tensor_for_vae


# ─────────────────────────────────────────────────────────────────────────────
# StreamingAvatarGenerator
# ─────────────────────────────────────────────────────────────────────────────

class StreamingAvatarGenerator:
    """
    Streaming talking-head video generator.

    Receives AudioFeatureChunk objects from the bridge and generates video
    frames block-by-block using AvatarForcing's sliding-window diffusion.

    Key invariants:
      - bridge output at 25 Hz → 1 audio feature frame ≈ 1 video frame
      - AvatarForcing generates num_frame_per_block frames per diffusion block
      - We wait until we have enough audio features for the next block before
        triggering diffusion, keeping latency low while maintaining sync

    Audio-Video Alignment:
        Moshi token @ 12.5 Hz → bridge × 2 → 25 Hz audio features
        AvatarForcing @ 25 fps → 1 audio feature per video frame ✓

    Args:
        pipeline: AvatarForcingInferencePipeline (already loaded)
        device: CUDA device
        dtype: computation dtype (bfloat16)
        num_frame_per_block: frames per diffusion block (from config, typically 4)
        prompt: text prompt for generation (fixed default prompt)
        audio_dim: dimension of audio features (9984 for wav2vec2 concat)
    """

    # Audio feature frames per video frame — bridge upsamples 12.5→25 Hz
    AUDIO_TO_VIDEO_RATIO = 1     # 1:1 after bridge upsampling (both 25 Hz)

    def __init__(
        self,
        pipeline,                          # AvatarForcingInferencePipeline
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        num_frame_per_block: int = 4,
        prompt: str = "",
        audio_dim: int = 10752,
        context_noise: float = 0.0,
        seed: int = 42,
    ):
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.num_frame_per_block = num_frame_per_block
        self.prompt = prompt
        self.audio_dim = audio_dim
        self.context_noise = context_noise

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # Latent shape (computed after first image encode)
        self._h: Optional[int] = None
        self._w: Optional[int] = None
        self._c: int = 16                  # VAE latent channels

        # Text embedding (pre-computed once from fixed prompt)
        self._prompt_cond: Optional[dict] = None

        # Initial image latent
        self._initial_latent: Optional[torch.Tensor] = None

        # KV caches (managed by pipeline — reset per conversation)
        self._frame_seq_length = pipeline.frame_seq_length   # 1560

        # Generation state
        self._generated_frames = 0        # total video frames generated
        self._prefix_frames = 0           # frames from image prefix (=1 for I2V)

        logger.info("StreamingAvatarGenerator initialized | block_size=%d frames", num_frame_per_block)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup_reference_image(self, image_tensor: torch.Tensor) -> None:
        """
        Encode reference image and initialize caches.

        image_tensor: (1, 3, 1, H, W) float — from load_reference_image()
        Must be called before any generation starts.
        """
        logger.info("Encoding reference image…")
        with torch.no_grad():
            # VAE encode: (1, 3, 1, H, W) → (1, 1, 16, H', W')
            initial_latent = self.pipeline.vae.encode_to_latent(
                image_tensor.to(device=self.device, dtype=self.dtype)
            )
        self._initial_latent = initial_latent              # (1, 1, 16, H', W')
        self._h = initial_latent.shape[-2]
        self._w = initial_latent.shape[-1]

        logger.info("Reference image encoded → latent %s", list(initial_latent.shape))

    def _prepare_text_prompt(self) -> None:
        """Pre-compute text embeddings from fixed prompt (called once)."""
        if self._prompt_cond is not None:
            return
        logger.info("Encoding text prompt…")
        with torch.no_grad():
            self._prompt_cond = self.pipeline.text_encoder(text_prompts=[self.prompt])
        logger.info("Text prompt encoded")

    def _build_y_conditioning(self, n_frames_total: int) -> torch.Tensor:
        """
        Build the image conditioning tensor y for I2V.
        y: (1, 17, T_total, H', W') — latent (16ch) + mask (1ch)
        """
        assert self._initial_latent is not None, "Call setup_reference_image() first"

        lat = self._initial_latent    # (1, 1, 16, H', W')
        img_lat = lat.permute(0, 2, 1, 3, 4)   # (1, 16, 1, H', W')

        T_total = n_frames_total + 20
        image_cat = img_lat.repeat(1, 1, T_total, 1, 1)  # (1, 16, T_total, H', W')
        msk = torch.zeros_like(image_cat[:, :1, :, :1, :1]).expand(
            -1, -1, -1, self._h, self._w
        ).clone()                               # (1, 1, T_total, H', W')
        msk[:, :, 1:] = 1.0                     # mask out all but first frame

        y = torch.cat([image_cat, msk], dim=1)  # (1, 17, T_total, H', W')
        return y.to(device=self.device, dtype=self.dtype)

    # ── Streaming generation ──────────────────────────────────────────────────

    async def generate_blocks(
        self,
        feature_buffer,                            # AudioFeatureBuffer
        feature_queue: asyncio.Queue,
        frame_queue: asyncio.Queue,
        total_blocks: int = 64,
        sentinel=None,
    ) -> None:
        """
        Main streaming generation coroutine.

        Waits for enough audio features in feature_buffer, then generates
        one block of video frames. Repeats until done or sentinel received.

        Args:
            feature_buffer: AudioFeatureBuffer filling from bridge
            feature_queue: receives AudioFeatureChunk from bridge streamer
            frame_queue: output queue for VideoFrame objects
            total_blocks: max blocks to generate (safety limit)
            sentinel: EOS sentinel value
        """
        assert self._initial_latent is not None, "Call setup_reference_image() first"

        self._prepare_text_prompt()

        # Reset pipeline caches for new conversation
        bsz = 1
        self.pipeline._reset_or_init_caches(
            batch_size=bsz, dtype=self.dtype, device=self.device
        )

        # Pre-fill KV cache with initial image frame
        # This mirrors _prefill_cache_for_rolling() for I2V
        output_latents = torch.zeros(
            [bsz, 1 + total_blocks * self.num_frame_per_block, self._c, self._h, self._w],
            device=self.device, dtype=self.dtype,
        )
        y_full = self._build_y_conditioning(total_blocks * self.num_frame_per_block)

        # Build conditional dict for the full sequence (text + y)
        # Audio features will be injected per-block
        base_cond = {}
        base_cond["prompt_embeds"] = self._prompt_cond["prompt_embeds"]

        # Prefill: process reference image frame (frame 0) through transformer
        zero_ts = torch.zeros([bsz, 1], device=self.device, dtype=torch.int64)
        output_latents[:, :1] = self._initial_latent

        # For the first frame, we need audio embedding covering at least 1 frame
        prefix_audio = torch.zeros(1, 1, self.audio_dim, device=self.device, dtype=self.dtype)
        cond_frame0 = {
            "prompt_embeds": base_cond["prompt_embeds"],
            "audio_emb": prefix_audio,
            "y": y_full[:, :, :1].contiguous(),
        }

        def _run_prefill():
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                    self.pipeline.generator(
                        noisy_image_or_video=self._initial_latent,
                        conditional_dict=cond_frame0,
                        timestep=zero_ts,
                        kv_cache=self.pipeline.kv_cache_clean,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=0,
                    )
        try:
            # Run synchronously on the main thread to prevent CUDA Graph conflicts
            _run_prefill()
            # Yield control so other tasks (like Moshi) can process buffered data
            await asyncio.sleep(0)
        except Exception as e:
            logger.error("Prefill step crashed: %s", e, exc_info=True)
            await frame_queue.put(sentinel)
            return

        self._prefix_frames = 1
        self._generated_frames = 0

        logger.info("KV cache prefilled with reference image frame")

        # ── Per-block streaming generation loop ───────────────────────────────

        t_pipeline_start = time.monotonic()
        eos_received = False

        for block_idx in range(total_blocks):
            # 1 latent video frame corresponds to 4 audio frames (due to VAE time stride of 4)
            audio_frames_needed = (block_idx + 1) * self.num_frame_per_block * 4

            # Wait until we have enough audio features for this block
            while not feature_buffer.has_enough(audio_frames_needed) and not eos_received:
                try:
                    chunk = await asyncio.wait_for(feature_queue.get(), timeout=0.5)
                    if chunk is sentinel:
                        eos_received = True
                        feature_queue.task_done()
                        break
                    feature_buffer.push(chunk)
                    feature_queue.task_done()
                except asyncio.TimeoutError:
                    if eos_received:
                        break

            # If EOS and not enough frames: generate with whatever we have
            audio_frames_available = feature_buffer.total_frames
            if audio_frames_available == 0:
                logger.info("No audio features available — stopping generation")
                break

            # Run one diffusion block synchronously on the main thread
            block_frames = self._generate_one_block(
                block_idx,
                output_latents,
                feature_buffer,
                y_full,
                base_cond,
                audio_frames_needed,
            )
            # Yield control so other tasks can process buffered data while we were blocking
            await asyncio.sleep(0)

            if block_frames is None:
                logger.warning("Block %d generation failed — stopping", block_idx)
                break

            # Emit decoded video frames
            t_emit = time.monotonic()
            latency_ms = (t_emit - t_pipeline_start) * 1000

            for fi, pixel_frame in enumerate(block_frames):
                global_fi = block_idx * self.num_frame_per_block + fi
                vf = VideoFrame(
                    pixels=pixel_frame,
                    frame_idx=global_fi,
                    timestamp=t_emit,
                    latency_ms=latency_ms,
                )
                await frame_queue.put(vf)
                logger.debug("Emitted frame %d | lat=%.1fms", global_fi, latency_ms)

            self._generated_frames += len(block_frames)

            if eos_received and audio_frames_available <= frames_needed:
                logger.info("EOS: generated %d total frames", self._generated_frames)
                break

        # Signal end of video stream
        await frame_queue.put(sentinel)
        logger.info("StreamingAvatarGenerator: generation complete")

    # ── Block-level diffusion (runs in thread pool) ───────────────────────────

    def _generate_one_block(
        self,
        block_idx: int,
        output_latents: torch.Tensor,
        feature_buffer,
        y_full: torch.Tensor,
        base_cond: dict,
        frames_needed: int,
    ) -> Optional[List[torch.Tensor]]:
        """
        Generate one block of num_frame_per_block video frames via AvatarForcing.

        This mirrors the per-window logic in inference_avatar_forcing() but
        operates one block at a time for streaming.

        Returns: list of (H, W, 3) uint8 pixel tensors, or None on error.
        """
        try:
            with torch.no_grad():
                # Frame indices for this block (0-indexed from prefix)
                frame_s = self._prefix_frames + block_idx * self.num_frame_per_block
                frame_e = frame_s + self.num_frame_per_block

                # Sample fresh noise for this block
                noise = torch.randn(
                    1, self.num_frame_per_block, self._c, self._h, self._w,
                    device=self.device, dtype=self.dtype,
                )

                # Slice audio features for this block
                # Each latent frame requires 4 audio frames (VAE time stride)
                # Prefix zero frame is at index 0
                audio_frames_per_block = self.num_frame_per_block * 4
                a_s = 1 + block_idx * audio_frames_per_block
                a_e = a_s + audio_frames_per_block
                
                audio_slice = feature_buffer.get_slice(a_s, a_e)
                audio_slice = audio_slice.to(device=self.device, dtype=self.dtype)

                # Slice y conditioning for this block
                y_slice = y_full[:, :, frame_s:frame_e].contiguous()

                # Build block conditional dict
                cond_block = {
                    "prompt_embeds": base_cond["prompt_embeds"],
                    "audio_emb": audio_slice,
                    "y": y_slice,
                }

                # AvatarForcing 1-step denoising for this streaming block
                noisy_input = noise.clone()
                denoise_steps = self.pipeline.denoise_steps.to(self.device)
                # We use the maximum timestep (e.g. 1000) for a 1-step generation from pure noise
                cur_timestep = torch.ones(
                    1, self.num_frame_per_block, device=self.device, dtype=torch.long
                ) * denoise_steps[0]

                with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_block,
                        timestep=cur_timestep,
                        kv_cache=self.pipeline.kv_cache_clean,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=frame_s * self._frame_seq_length,
                    )

                # Store in output latent buffer
                output_latents[:, frame_s:frame_e] = denoised_pred

                # Update KV cache (context noise injection for temporal consistency)
                context_noise = self.context_noise
                cache_ts = torch.ones(
                    1, self.num_frame_per_block, device=self.device, dtype=torch.int64
                ) * int(context_noise)
                cache_lat = denoised_pred

                self.pipeline.generator(
                    noisy_image_or_video=cache_lat,
                    conditional_dict=cond_block,
                    timestep=cache_ts,
                    kv_cache=self.pipeline.kv_cache_clean,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=frame_s * self._frame_seq_length,
                    updating_cache=True,
                )

                # VAE decode: latents → pixels
                pixel_block = self._decode_latents(denoised_pred)    # (1, T, 3, H, W)
            
            # Convert to list of (H, W, 3) uint8
            frames = []
            for fi in range(pixel_block.shape[1]):
                px = pixel_block[0, fi]                          # (3, H, W) float [0,1]
                px_uint8 = (px * 255).clamp(0, 255).to(torch.uint8)
                px_hwc = px_uint8.permute(1, 2, 0)              # (H, W, 3)
                frames.append(px_hwc.cpu())

            return frames

        except Exception as e:
            logger.error("Block %d generation error: %s", block_idx, e, exc_info=True)
            return None

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to pixel space via the WAN VAE.
        latents: (1, T, 16, H', W') → pixels: (1, T, 3, H, W) float [0,1]
        """
        # VAE decode
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            pixels = self.pipeline.vae.decode_to_pixel(latents, use_cache=False)
        # Normalize from [-1,1] → [0,1]
        pixels = (pixels * 0.5 + 0.5).clamp(0.0, 1.0)
        self.pipeline.vae.model.clear_cache()
        return pixels
