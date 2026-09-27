"""
tracker.py
----------
Thin, typed wrapper around the SORT algorithm (sort/sort.py).
"""

import logging

import numpy as np

from sort.sort import Sort

logger = logging.getLogger("object_tracking")


class Tracker:
    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        logger.info(
            f"Tracker initialized (max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold})"
        )

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        detections: (N, 6) -> [x1, y1, x2, y2, conf, class_id]
        Returns: (M, 7) -> [x1, y1, x2, y2, track_id, conf, class_id]
        """
        return self.tracker.update(detections)