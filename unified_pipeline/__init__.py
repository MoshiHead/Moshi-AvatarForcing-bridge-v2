"""
Unified Real-Time Streaming Pipeline
Moshi Speech Foundation Model → Bridge → AvatarForcing Talking-Head Generation

Architecture:
    User Audio → Moshi LM (discrete tokens) → Bridge (wav2vec-like features)
                → AvatarForcing (streaming talking-head frames) → Real-Time Video
"""

__version__ = "1.0.0"
__author__ = "Moshi-AvatarForcing Unified Pipeline"
