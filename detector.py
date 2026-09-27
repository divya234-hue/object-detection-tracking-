"""
detector.py
-----------
Wraps a pre-trained YOLOv8 model and turns raw model output into a clean,
SORT-ready detection format: [x1, y1, x2, y2, confidence, class_id]
"""

import logging
from typing import Dict

import numpy as np
from ultralytics import YOLO

from exceptions import ModelLoadError, DetectionError

logger = logging.getLogger("object_tracking")


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold

        try:
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YOLO model '{model_path}'. "
                f"Check internet connection (first run downloads weights) "
                f"and that ultralytics is installed correctly. Original error: {e}"
            )

        self.class_names: Dict[int, str] = self.model.names
        logger.info(f"Model loaded. {len(self.class_names)} classes available.")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns np.ndarray (N, 6) -> [x1, y1, x2, y2, confidence, class_id]
        """
        if frame is None:
            raise DetectionError("Invalid frame passed to detector (frame is None).")

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as e:
            raise DetectionError(f"YOLO inference failed: {e}")

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                if confidence >= self.confidence_threshold:
                    detections.append([x1, y1, x2, y2, confidence, class_id])

        if len(detections) == 0:
            return np.empty((0, 6))

        return np.array(detections)

    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(int(class_id), "unknown")