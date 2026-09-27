"""
utils.py
--------
Drawing helpers + JSON track-history logging.
"""

import json
import logging
import time
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger("object_tracking")

_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def get_color(class_id: int):
    return _COLORS[int(class_id) % len(_COLORS)]


def draw_tracked_objects(frame: np.ndarray, tracked_objects: np.ndarray, class_names: dict) -> np.ndarray:
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, conf, class_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id, class_id = int(track_id), int(class_id)

        class_name = class_names.get(class_id, "object")
        color = get_color(class_id)
        label = f"{class_name} | ID:{track_id} | {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


class TrackLogger:
    """
    Accumulates per-frame tracking results and writes them to a JSON file
    on demand — useful for later analysis, audit trails, or debugging
    in a production setting instead of only visual output.
    """

    def __init__(self, output_path: str, class_names: dict):
        self.output_path = output_path
        self.class_names = class_names
        self.records: List[Dict] = []
        self.frame_index = 0

    def log_frame(self, tracked_objects: np.ndarray) -> None:
        objects = []
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, conf, class_id = obj
            objects.append({
                "track_id": int(track_id),
                "class_name": self.class_names.get(int(class_id), "unknown"),
                "confidence": round(float(conf), 3),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
            })

        self.records.append({
            "frame": self.frame_index,
            "timestamp": time.time(),
            "objects": objects,
        })
        self.frame_index += 1

    def save(self) -> None:
        try:
            with open(self.output_path, "w") as f:
                json.dump(self.records, f, indent=2)
            logger.info(f"Track log saved to {self.output_path} ({len(self.records)} frames)")
        except OSError as e:
            logger.error(f"Failed to save track log: {e}")