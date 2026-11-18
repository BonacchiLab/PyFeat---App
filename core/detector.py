"""Py-Feat detector lifecycle and inference helpers."""

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

import torch
from feat import Detector
import feat.detector as feat_detector

_INFERENCE_LOCK = Lock()

@lru_cache(maxsize=1)
def load_detector() -> Detector:
    return Detector(device="cuda" if torch.cuda.is_available() else "cpu")


def detect_video(video_path: Path, skip_frames: int, progress_callback: Optional[Callable] = None):
    """Run inference under no_grad, optionally reporting Py-Feat batch progress."""
    with _INFERENCE_LOCK:
        with torch.no_grad():
            if progress_callback is None:
                return load_detector().detect_video(str(video_path), skip_frames=skip_frames, batch_size=4)
            original_tqdm = feat_detector.tqdm
            feat_detector.tqdm = lambda iterable: _progress_iterator(iterable, progress_callback)
            try:
                return load_detector().detect_video(str(video_path), skip_frames=skip_frames, batch_size=4)
            finally:
                feat_detector.tqdm = original_tqdm


def _progress_iterator(iterable, callback):
    total = len(iterable)
    for number, item in enumerate(iterable, start=1):
        yield item
        callback(number, total)
