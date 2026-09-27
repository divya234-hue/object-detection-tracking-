import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.4):
        """
        model_path: name/path of the YOLO weights file.
                    "yolov8n.pt" is auto-downloaded by ultralytics on first run.
        confidence_threshold: detections below this score are discarded.
        """
        self.confidence_threshold = confidence_threshold

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_path}'. "
                f"Check your internet connection (first run needs to download "
                f"the weights) and that ultralytics is installed correctly. "
                f"Original error: {e}"
            )

        # class_id -> class_name, e.g. {0: 'person', 2: 'car', ...}
        self.class_names = self.model.names

    def detect(self, frame):
        if frame is None:
            raise ValueError("Invalid frame passed to detector (frame is None).")

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

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

    def get_class_name(self, class_id):
        """Converts a numeric class_id into a readable name, e.g. 'person'."""
        return self.class_names.get(int(class_id), "unknown")