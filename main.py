import os
import time
import cv2

from detector import Detector
from tracker import Tracker
from utils import draw_tracked_objects, draw_fps

# --------------------------------------------------------------------------
# CONFIGURATION - change these values to customize behavior
# --------------------------------------------------------------------------

# Set to 0 (int) for webcam, or "video.mp4" (string path) for a video file.
VIDEO_SOURCE = 0

MODEL_PATH = "yolov8n.pt"       # swap to "yolo11n.pt" for the newer model
CONFIDENCE_THRESHOLD = 0.4      # detections below this score are ignored
RESIZE_WIDTH = 640              # set to None to disable resizing

SAVE_DIR = "saved_frames"       # where 's' key saves snapshots

# --------------------------------------------------------------------------


def open_video_source(source):
    """Opens webcam (int) or video file (str) and validates it opened correctly."""
    if isinstance(source, str) and not os.path.exists(source):
        raise FileNotFoundError(f"Video file not found: '{source}'")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        if source == 0 or isinstance(source, int):
            raise RuntimeError("Could not open webcam. Check that it's connected and not used by another app.")
        else:
            raise RuntimeError(f"Could not open video file: '{source}'")

    return cap


def resize_frame(frame, target_width):
    """Resizes frame to target_width, keeping aspect ratio, to speed up CPU inference."""
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h))


def main():
    print("Loading YOLO model...")
    try:
        detector = Detector(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    tracker = Tracker(max_age=15, min_hits=3, iou_threshold=0.3)

    try:
        cap = open_video_source(VIDEO_SOURCE)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("Starting video stream. Press 'q' to quit, 's' to save a frame.")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[INFO] End of video stream / could not read frame.")
            break

        frame = resize_frame(frame, RESIZE_WIDTH)

        try:
            detections = detector.detect(frame)
        except ValueError as e:
            print(f"[WARNING] Skipping invalid frame: {e}")
            continue

        # detections may be empty -> tracker.update handles that gracefully
        tracked_objects = tracker.update(detections)

        frame = draw_tracked_objects(frame, tracked_objects, detector.class_names)

        # FPS calculation
        frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = curr_time
        else:
            fps = frame_count / elapsed if elapsed > 0 else 0.0

        frame = draw_fps(frame, fps)

        cv2.imshow("Object Detection and Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed. Exiting...")
            break
        elif key == ord("s"):
            filename = os.path.join(SAVE_DIR, f"frame_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Frame saved to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()