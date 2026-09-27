"""
video_source.py
----------------
Wraps OpenCV's VideoCapture with proper validation, error handling,
and a clean context-manager interface (use with `with` for guaranteed
resource cleanup, even on crash).
"""

import logging
import os
from typing import Optional, Union

import cv2
import numpy as np

from exceptions import VideoSourceError

logger = logging.getLogger("object_tracking")


class VideoSource:
    def __init__(self, source: Union[int, str]):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def open(self) -> None:
        if isinstance(self.source, str) and not os.path.exists(self.source):
            raise VideoSourceError(f"Video file not found: '{self.source}'")

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            if self.source == 0 or isinstance(self.source, int):
                raise VideoSourceError(
                    "Could not open webcam. Check that it's connected and not used by another app."
                )
            raise VideoSourceError(f"Could not open video file: '{self.source}'")

        logger.info(f"Video source opened: {self.source}")

    def read(self):
        if self.cap is None:
            raise VideoSourceError("Video source not opened. Call open() first.")
        ret, frame = self.cap.read()
        return ret, frame

    def get_fps(self) -> float:
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("Video source released.")
            self.cap = None


def resize_frame(frame: np.ndarray, target_width: Optional[int]) -> np.ndarray:
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h))