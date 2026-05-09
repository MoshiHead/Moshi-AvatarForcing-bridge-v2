"""
bridge_dataset.py — Bridge-Injected Dataset for AvatarForcing
==============================================================
Drop-in replacement for TextImageAudioPairDataset that BYPASSES wav2vec.

Instead of encoding audio through wav2vec2, this dataset accepts
pre-computed bridge features (from MimiWav2Vec2Bridge) directly.

Use this class when running the unified offline batch pipeline where
you pre-process audio through the bridge first.

For streaming (real-time) mode, features are injected directly by
the async_pipeline.py orchestrator — this class is not needed.
"""

import torch
import math
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F_tv
import shlex


class ResizeKeepRatioArea16:
    def __init__(self, area_hw=(480, 832), div=16):
        self.A = area_hw[0] * area_hw[1]
        self.d = div

    def __call__(self, img):
        w, h = img.size
        s = min(1.0, math.sqrt(self.A / (h * w)))
        nh = max(self.d, int(h * s) // self.d * self.d)
        nw = max(self.d, int(w * s) // self.d * self.d)
        return F_tv.resize(img, (nh, nw), interpolation=InterpolationMode.BILINEAR, antialias=True)


class BridgeInjectedDataset(Dataset):
    """
    Dataset for AvatarForcing that uses pre-computed bridge features
    instead of wav2vec embeddings.

    Data file format (tab/space-separated, one entry per line):
        <image_path> <bridge_feature_path.pt> <caption>

    Where <bridge_feature_path.pt> is a saved tensor of shape
        (T, audio_dim)  — e.g. (25*duration, 9984)
    saved with torch.save().

    These can be pre-generated with:
        python precompute_bridge_features.py --audio <dir> --out <dir>
    """

    DEFAULT_PROMPT = (
        "A realistic person speaking naturally with accurate lip synchronization, "
        "expressive facial motion, stable head movement, and smooth real-time talking animation."
    )

    def __init__(
        self,
        path: str,
        target_area=(832, 480),
        teacher_len: int = 21,
        fps: int = 25,
        transform=None,
        max_samples=None,
        audio_dim: int = 9984,
        override_prompt: str = None,
    ):
        self.target_area = tuple(target_area)
        self.teacher_len = teacher_len
        self.fps = fps
        self.audio_dim = audio_dim
        self.override_prompt = override_prompt or self.DEFAULT_PROMPT
        self.transform = transform or transforms.Compose([
            ResizeKeepRatioArea16((480, 832), 16),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if max_samples:
            lines = lines[:max_samples]

        self.data = []
        for ln in lines:
            parts = shlex.split(ln)
            if len(parts) == 3:
                img, feat, cap = parts
            elif len(parts) == 2:
                img, feat = parts
                cap = self.override_prompt
            else:
                continue
            self.data.append({
                "image_path": img,
                "feature_path": feat,
                "caption": cap,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Load image
        image = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(image)   # (3, H, W)

        # Load pre-computed bridge features
        feat_path = row["feature_path"]
        if not Path(feat_path).exists():
            return None

        audio_emb = torch.load(feat_path, map_location="cpu")   # (T, audio_dim)
        if audio_emb.dim() == 3:
            audio_emb = audio_emb.squeeze(0)   # (T, audio_dim)

        # Trim/pad to teacher_len
        if audio_emb.shape[0] < self.teacher_len:
            return None   # too short

        audio_emb = audio_emb[:self.teacher_len]   # (teacher_len, audio_dim)

        # Add prefix zero frame (matches original AvatarForcing dataset convention)
        prefix = torch.zeros(1, self.audio_dim, dtype=audio_emb.dtype)
        audio_emb = torch.cat([prefix, audio_emb], dim=0)   # (teacher_len+1, audio_dim)

        # Override with fixed prompt
        caption = self.override_prompt

        return {
            "image": img,
            "audio_emb": audio_emb,
            "prompts": [caption],
            "idx": idx,
        }


class SingleShotBridgeDataset(Dataset):
    """
    Minimal single-item dataset for one-shot inference.
    Takes a pre-loaded image tensor and audio feature tensor directly.
    Used by the streaming pipeline for the first-pass warmup.
    """

    def __init__(
        self,
        image_tensor: torch.Tensor,       # (3, H, W)
        audio_features: torch.Tensor,     # (T, audio_dim) — from bridge
        prompt: str = None,
    ):
        self.image_tensor = image_tensor
        self.audio_features = audio_features
        self.prompt = prompt or BridgeInjectedDataset.DEFAULT_PROMPT

        # Add prefix zero frame
        prefix = torch.zeros(1, audio_features.shape[-1], dtype=audio_features.dtype)
        self.audio_emb = torch.cat([prefix, audio_features], dim=0)   # (T+1, dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "image": self.image_tensor,
            "audio_emb": self.audio_emb,
            "prompts": [self.prompt],
            "idx": 0,
        }
