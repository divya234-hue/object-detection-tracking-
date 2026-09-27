import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def bbox_to_state(bbox):
    """
    Converts [x1, y1, x2, y2] into the Kalman filter's measurement format:
    [center_x, center_y, area, aspect_ratio]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    area = width * height
    aspect_ratio = width / float(height) if height != 0 else 0
    return np.array([center_x, center_y, area, aspect_ratio]).reshape((4, 1))


def state_to_bbox(state):
    """
    Converts the Kalman filter's state [cx, cy, area, aspect_ratio, ...]
    back into a bounding box [x1, y1, x2, y2].
    """
    cx, cy, area, aspect_ratio = state[0], state[1], state[2], state[3]
    area = max(area, 1e-6)
    width = np.sqrt(area * aspect_ratio)
    height = area / width if width != 0 else 0
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


def iou_batch(boxes_a, boxes_b):
    """
    Computes IoU (Intersection over Union) between every box in boxes_a
    and every box in boxes_b.

    IoU = overlap area / combined area, ranges 0 (no overlap) to 1 (identical).
    Used to decide if a predicted box and a detected box are "the same object".

    Returns a matrix of shape (len(boxes_a), len(boxes_b)).
    """
    boxes_a = np.expand_dims(boxes_a, 1)
    boxes_b = np.expand_dims(boxes_b, 0)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    intersection = inter_w * inter_h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - intersection

    return np.where(union > 0, intersection / union, 0.0)


class KalmanBoxTracker:
    """
    Represents a single tracked object.
    Wraps a Kalman Filter that predicts and updates the object's bounding box.
    """
    count = 0  # class-level counter used to hand out unique IDs

    def __init__(self, bbox):
        # 7 state variables: cx, cy, area, aspect_ratio, vx, vy, v_area
        # 4 measurement variables: cx, cy, area, aspect_ratio
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity motion model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement function (we only directly observe cx, cy, area, ratio)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Uncertainty tuning (standard, beginner-safe defaults)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_state(bbox)

        self.time_since_update = 0

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count  # unique tracking ID

        self.hits = 0            # total number of successful matches
        self.hit_streak = 0      # consecutive frames matched
        self.age = 0              # total frames since creation

    def update(self, bbox):
        """Corrects the tracker's state using a newly matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_state(bbox))

    def predict(self):
        """Advances the state one frame forward and returns the predicted box."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return state_to_bbox(self.kf.x)

    def get_state(self):
        """Returns the current best estimate of the bounding box."""
        return state_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Matches new detections to existing trackers using IoU + Hungarian algorithm.

    Returns:
        matches            -> list of (detection_index, tracker_index)
        unmatched_dets     -> detections with no matching tracker (new objects)
        unmatched_trackers  -> trackers with no matching detection (lost objects)
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_detections.append(det_idx)
            unmatched_trackers.append(trk_idx)
        else:
            matches.append((det_idx, trk_idx))

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    The main SORT tracker. Call update() once per frame with the latest
    YOLO detections; it returns tracked objects with stable IDs.
    """

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        """
        max_age: how many frames a tracker is kept alive without a match
                 before it's deleted (object considered "gone").
        min_hits: how many consecutive matches a tracker needs before it's
                  considered confirmed (reduces flicker on noisy detections).
        iou_threshold: minimum IoU to consider a detection and tracker a match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        detections: np.ndarray of shape (N, 6) -> [x1, y1, x2, y2, conf, class_id]
                    (can be empty)

        Returns: np.ndarray of shape (M, 7)
                 -> [x1, y1, x2, y2, track_id, conf, class_id]
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6))

        det_boxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))

        # Step 1: predict new locations for all existing trackers
        predicted_boxes = []
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()[0]
            if np.any(np.isnan(pred)):
                to_delete.append(i)
            else:
                predicted_boxes.append(pred)

        for i in reversed(to_delete):
            self.trackers.pop(i)

        predicted_boxes = np.array(predicted_boxes) if len(predicted_boxes) > 0 else np.empty((0, 4))
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, predicted_boxes, self.iou_threshold
        )
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(det_boxes[det_idx])
        for det_idx in unmatched_dets:
            new_tracker = KalmanBoxTracker(det_boxes[det_idx])
            self.trackers.append(new_tracker)
        results = []
        matched_det_for_tracker = {trk_idx: det_idx for det_idx, trk_idx in matches}

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            bbox = trk.get_state()[0]

            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or trk.age <= self.min_hits):
                # find the detection's confidence/class if this tracker was just matched
                conf, class_id = 0.0, -1
                if i in matched_det_for_tracker:
                    det_idx = matched_det_for_tracker[i]
                    conf = detections[det_idx, 4]
                    class_id = detections[det_idx, 5]

                results.append(np.concatenate((bbox, [trk.id], [conf], [class_id])))

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(results) > 0:
            return np.array(results)
        return np.empty((0, 7))