"""
run_pipeline.py — CLI entry point for the unified pipeline.
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import torch

# ── Ensure repo roots are on path ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
for sub in ["moshi-inference", "AvatarForcing-inference", "bridge_module"]:
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from unified_pipeline.config import PipelineConfig, MoshiConfig, BridgeConfig, AvatarForcingConfig
from unified_pipeline.model_loader import load_all_models
from unified_pipeline.async_pipeline import UnifiedPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Moshi → Bridge → AvatarForcing streaming pipeline")
    p.add_argument("--image", required=True, help="Reference portrait image path")
    p.add_argument("--audio", required=True, help="Input audio WAV path (24kHz mono)")
    p.add_argument("--output", default="output_avatar.mp4", help="Output video path")
    p.add_argument("--moshi-repo", default="kyutai/moshiko-pytorch-bf16")
    p.add_argument("--bridge-ckpt", required=True, help="Bridge checkpoint .pt path")
    p.add_argument("--bridge-cfg", required=True, help="Bridge config.yaml path")
    p.add_argument("--af-ckpt", required=True, help="AvatarForcing checkpoint .pt path")
    p.add_argument("--af-cfg", default="AvatarForcing-inference/configs/avatarforcing.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--max-blocks", type=int, default=64)
    p.add_argument("--chunk-tokens", type=int, default=4,
                   help="Moshi token frames per bridge chunk (latency vs. throughput)")
    p.add_argument("--no-projection", action="store_true",
                   help="Skip 768→9984 projection (only if WanModel accepts 768-dim)")
    p.add_argument("--serve", action="store_true", help="Start WebSocket server instead of batch mode")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


async def main_async(args):
    # ── Build config ──────────────────────────────────────────────────────────
    cfg = PipelineConfig(
        moshi=MoshiConfig(
            hf_repo=args.moshi_repo,
            device=args.device,
        ),
        bridge=BridgeConfig(
            checkpoint_path=args.bridge_ckpt,
            config_path=args.bridge_cfg,
            device=args.device,
            chunk_tokens=args.chunk_tokens,
            use_projection=not args.no_projection,
        ),
        avatar=AvatarForcingConfig(
            config_path=args.af_cfg,
            checkpoint_path=args.af_ckpt,
            device=args.device,
            fps=args.fps,
        ),
        port=args.port,
    )

    # ── Load all models ───────────────────────────────────────────────────────
    logger.info("Loading all models…")
    models = load_all_models(cfg)

    # ── Build pipeline ────────────────────────────────────────────────────────
    unified = UnifiedPipeline(cfg, models)
    unified.setup_reference_image(args.image)

    if args.serve:
        # WebSocket server mode
        from unified_pipeline.async_pipeline import StreamingPipelineServer
        server = StreamingPipelineServer(unified)
        await server.serve(host="0.0.0.0", port=args.port)
        return

    # ── Batch mode: process audio file ────────────────────────────────────────
    import sphn
    logger.info("Loading audio: %s", args.audio)
    in_pcms, _ = sphn.read(args.audio, sample_rate=models["mimi"].sample_rate)
    import torch
    audio_tensor = torch.from_numpy(in_pcms).to(args.device)
    audio_tensor = audio_tensor[None, 0:1]   # [1, 1, N]

    logger.info("Starting pipeline…")
    out_path = await unified.run_to_video(
        audio_tensor,
        output_path=args.output,
        fps=args.fps,
        max_blocks=args.max_blocks,
    )
    logger.info("Done → %s", out_path)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
