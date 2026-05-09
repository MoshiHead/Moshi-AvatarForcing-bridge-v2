"""
precompute_bridge_features.py — Offline bridge feature extraction
=================================================================
Pre-computes bridge audio features from audio files to avoid runtime
overhead when running AvatarForcing in batch (non-streaming) mode.

Usage:
    python precompute_bridge_features.py \\
        --audio-dir /path/to/wavs \\
        --out-dir /path/to/features \\
        --bridge-ckpt checkpoints/bridge_best.pt \\
        --bridge-cfg bridge_module/config.yaml \\
        --mimi-repo kyutai/moshiko-pytorch-bf16

Output: For each <name>.wav, saves <name>.pt with shape (T, 9984).
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bridge-ckpt", required=True)
    p.add_argument("--bridge-cfg", required=True)
    p.add_argument("--moshi-root", default="moshi-inference")
    p.add_argument("--bridge-root", default="bridge_module")
    p.add_argument("--mimi-repo", default="kyutai/moshiko-pytorch-bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-tokens", type=int, default=8)
    p.add_argument("--upsample-factor", type=int, default=2)
    p.add_argument("--target-fps", type=float, default=25.0)
    p.add_argument("--ext", default=".wav", help="Audio file extension to scan")
    return p.parse_args()


def load_mimi(moshi_root, mimi_repo, device):
    sys.path.insert(0, moshi_root)
    from moshi.models import loaders
    ckpt = loaders.CheckpointInfo.from_hf_repo(mimi_repo)
    mimi = ckpt.get_mimi(device=device)
    mimi.eval()
    return mimi


def load_bridge(bridge_root, ckpt_path, cfg_path, device):
    sys.path.insert(0, bridge_root)
    from model import MimiWav2Vec2Bridge
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    bridge = MimiWav2Vec2Bridge(cfg)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("bridge", raw.get("state_dict", raw))
    bridge.load_state_dict(sd, strict=True)
    bridge = bridge.to(device=device, dtype=torch.bfloat16).eval()
    bridge.requires_grad_(False)
    return bridge, cfg


def load_projection(ckpt_path, in_dim=768, out_dim=9984, device="cuda"):
    """Load or init the BridgeAudioProjection."""
    import torch.nn as nn

    class Proj(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.proj = nn.Linear(i, o)
            self.norm = nn.LayerNorm(o)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        def forward(self, x):
            return self.norm(self.proj(x))

    proj = Proj(in_dim, out_dim)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "projection" in raw:
        proj.load_state_dict(raw["projection"])
        logger.info("Projection weights loaded from checkpoint")
    else:
        logger.warning("No projection weights found — using random init")
    return proj.to(device=device, dtype=torch.bfloat16).eval()


@torch.no_grad()
def encode_audio_file(audio_path, mimi, bridge, projection, device, chunk_size=8, upsample=2):
    """
    Encode one audio file through Mimi → Bridge → Projection.
    Returns feature tensor of shape (T_out, 9984).
    """
    import sphn
    pcms, _ = sphn.read(str(audio_path), sample_rate=mimi.sample_rate)
    pcm_tensor = torch.from_numpy(pcms).to(device)[None, 0:1]  # [1, 1, N]

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    chunks = [c for c in pcm_tensor.split(frame_size, dim=2) if c.shape[-1] == frame_size]

    token_buf = []
    feature_buf = []
    kv_cache = None

    for chunk in chunks:
        codes = mimi.encode(chunk)          # [1, n_q, 1]
        tokens_1d = codes[0, :8, 0]        # (8,) — first 8 codebooks (dep_q)
        token_buf.append(tokens_1d)

        if len(token_buf) >= chunk_size:
            batch = torch.stack(token_buf).unsqueeze(0)   # (1, chunk_size, 8)
            feats, kv_cache = bridge(batch, use_cache=True, past_kvs=kv_cache)
            proj_feats = projection(feats)                # (1, chunk_size*upsample, 9984)
            feature_buf.append(proj_feats.squeeze(0).float().cpu())
            token_buf.clear()

    # Flush remainder
    if token_buf:
        batch = torch.stack(token_buf).unsqueeze(0)
        feats, _ = bridge(batch, use_cache=True, past_kvs=kv_cache)
        proj_feats = projection(feats)
        feature_buf.append(proj_feats.squeeze(0).float().cpu())

    if not feature_buf:
        return None

    return torch.cat(feature_buf, dim=0)   # (T_total, 9984)


def main():
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Mimi…")
    mimi = load_mimi(args.moshi_root, args.mimi_repo, device)

    logger.info("Loading Bridge…")
    bridge, bridge_cfg = load_bridge(args.bridge_root, args.bridge_ckpt, args.bridge_cfg, device)

    out_dim = bridge_cfg["model"]["output_dim"]    # 768
    target_dim = out_dim * 13                      # 9984
    logger.info("Loading Projection (%d→%d)…", out_dim, target_dim)
    projection = load_projection(args.bridge_ckpt, out_dim, target_dim, device)

    audio_files = sorted(Path(args.audio_dir).glob(f"*{args.ext}"))
    logger.info("Found %d audio files in %s", len(audio_files), args.audio_dir)

    mimi.streaming_forever(1)

    for i, af in enumerate(audio_files):
        out_path = out_dir / (af.stem + ".pt")
        if out_path.exists():
            logger.info("[%d/%d] SKIP (exists): %s", i+1, len(audio_files), af.name)
            continue

        logger.info("[%d/%d] Processing: %s", i+1, len(audio_files), af.name)
        try:
            feats = encode_audio_file(
                af, mimi, bridge, projection, device,
                chunk_size=args.chunk_tokens,
                upsample=args.upsample_factor,
            )
            if feats is None:
                logger.warning("  Skipped — no features generated")
                continue

            torch.save(feats, out_path)
            logger.info("  Saved: %s | shape=%s", out_path.name, list(feats.shape))

        except Exception as e:
            logger.error("  ERROR: %s", e, exc_info=True)

    logger.info("Done — processed %d files", len(audio_files))


if __name__ == "__main__":
    main()
