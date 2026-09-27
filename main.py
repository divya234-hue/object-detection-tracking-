"""
main.py
-------
Entry point: loads config, sets up logging, runs the detection +
tracking pipeline, optionally saves annotated video output and a
JSON log of all tracked objects.
"""

import os
import time

import cv2

from config import load_config, parse_cli_args, apply_cli_overrides
from detector import Detector
from exceptions import VideoSourceError, ModelLoadError, DetectionError, ConfigError
from logger_setup import setup_logger
from tracker import Tracker
from utils import draw_tracked_objects, draw_fps, TrackLogger
from video_source import VideoSource, resize_frame


def main():
    args = parse_cli_args()

    try:
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        return

    logger = setup_logger(log_dir=config.logging.log_dir)
    logger.info("Starting Object Detection and Tracking application")
    logger.debug(f"Loaded config: {config}")

    try:
        detector = Detector(
            model_path=config.detection.model_path,
            confidence_threshold=config.detection.confidence_threshold,
        )
    except ModelLoadError as e:
        logger.error(str(e))
        return

    tracker = Tracker(
        max_age=config.tracking.max_age,
        min_hits=config.tracking.min_hits,
        iou_threshold=config.tracking.iou_threshold,
    )

    track_logger = None
    if config.logging.save_tracks_json:
        track_logger = TrackLogger(config.logging.tracks_json_path, detector.class_names)

    video_writer = None

    try:
        with VideoSource(config.video.source) as video:
            if config.video.save_output:
                os.makedirs(os.path.dirname(config.video.output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # dimensions determined after first frame is read

            prev_time = time.time()
            frame_count = 0
            fps_display = 0.0

            logger.info("Press 'q' to quit, 's' to save a snapshot.")

            while True:
                ret, frame = video.read()
                if not ret or frame is None:
                    logger.info("End of video stream / could not read frame.")
                    break

                frame = resize_frame(frame, config.video.resize_width)

                if config.video.save_output and video_writer is None:
                    h, w = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        config.video.output_path, fourcc, 20.0, (w, h)
                    )
                    logger.info(f"Saving output video to {config.video.output_path}")

                try:
                    detections = detector.detect(frame)
                except DetectionError as e:
                    logger.warning(f"Skipping frame due to detection error: {e}")
                    continue

                tracked_objects = tracker.update(detections)

                if track_logger is not None:
                    track_logger.log_frame(tracked_objects)

                frame = draw_tracked_objects(frame, tracked_objects, detector.class_names)

                frame_count += 1
                elapsed = time.time() - prev_time
                if elapsed >= 1.0:
                    fps_display = frame_count / elapsed
                    frame_count = 0
                    prev_time = time.time()

                frame = draw_fps(frame, fps_display)

                if video_writer is not None:
                    video_writer.write(frame)

                if not args.no_display:
                    cv2.imshow("Object Detection and Tracking", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("'q' pressed. Exiting...")
                        break
                    elif key == ord("s"):
                        snap_path = f"outputs/snapshot_{int(time.time())}.jpg"
                        os.makedirs("outputs", exist_ok=True)
                        cv2.imwrite(snap_path, frame)
                        logger.info(f"Snapshot saved to {snap_path}")

    except VideoSourceError as e:
        logger.error(str(e))
        return
    finally:
        if video_writer is not None:
            video_writer.release()
        if track_logger is not None:
            track_logger.save()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete. Application exited.")


if __name__ == "__main__":
    main()