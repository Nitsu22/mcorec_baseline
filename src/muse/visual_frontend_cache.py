import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset.avhubert_dataset import load_video


_MUSE_NORM_MEAN = 0.4161
_MUSE_NORM_STD = 0.1688
_MUSE_ROI_SIZE = 112


def _as_str(path):
    if isinstance(path, bytes):
        return os.fsdecode(path)
    return path


def _make_cache_key(video_path: str, start_time: Optional[float], end_time: Optional[float]) -> str:
    key = f"{video_path}|{start_time if start_time is not None else ''}|{end_time if end_time is not None else ''}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class MuSEFeatureCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def key(self, video_path: str, start_time: Optional[float], end_time: Optional[float]) -> str:
        return _make_cache_key(_as_str(video_path), start_time, end_time)

    def path(self, key: str) -> str:
        subdir = os.path.join(self.cache_dir, key[:2])
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{key}.npy")

    def exists(self, video_path: str, start_time: Optional[float], end_time: Optional[float]) -> bool:
        key = self.key(video_path, start_time, end_time)
        return os.path.exists(self.path(key))

    def load(self, video_path: str, start_time: Optional[float], end_time: Optional[float]) -> np.ndarray:
        key = self.key(video_path, start_time, end_time)
        path = self.path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MuSE cache miss: {path}")
        return np.load(path)

    def save(
        self,
        video_path: str,
        start_time: Optional[float],
        end_time: Optional[float],
        features: np.ndarray,
    ) -> str:
        key = self.key(video_path, start_time, end_time)
        path = self.path(key)
        tmp_path = f"{path}.tmp.npy"
        np.save(tmp_path, features)
        os.replace(tmp_path, path)
        return path


@dataclass
class VisualFrontendExtractor:
    checkpoint_path: str
    device: str = "cpu"

    def __post_init__(self):
        muse_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "MuSE")
        )
        if muse_root not in sys.path:
            sys.path.append(muse_root)
        from pretrain_networks.visual_frontend import VisualFrontend  # type: ignore

        self.device = torch.device(self.device)
        self.model = VisualFrontend()
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def extract(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        video_path = _as_str(video_path)
        video = load_video(video_path, start_time=start_time or 0, end_time=end_time)
        # video: T x 1 x H x W (uint8)
        video = video.float() / 255.0
        if video.shape[-1] != _MUSE_ROI_SIZE or video.shape[-2] != _MUSE_ROI_SIZE:
            video = F.interpolate(video, size=(_MUSE_ROI_SIZE, _MUSE_ROI_SIZE), mode="bilinear", align_corners=False)
        video = (video - _MUSE_NORM_MEAN) / _MUSE_NORM_STD
        # VisualFrontend expects: T x 1 x 1 x H x W
        video = video.unsqueeze(1).to(self.device)
        with torch.no_grad():
            feats = self.model(video)
        feats = feats.squeeze(1).cpu().numpy().astype(np.float32)
        return feats
