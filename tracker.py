from sort.sort import Sort


class Tracker:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections):
        """
        detections: np.ndarray (N, 6) -> [x1, y1, x2, y2, conf, class_id]

        Returns: np.ndarray (M, 7) -> [x1, y1, x2, y2, track_id, conf, class_id]
        """
        return self.tracker.update(detections)