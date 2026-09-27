"""
config.py
---------
Loads settings from config.yaml into a typed dataclass, with optional
CLI argument overrides. This replaces hardcoded constants scattered
across files — industry projects centralize configuration.
"""

import argparse
import os
from dataclasses import dataclass, field

import yaml

from exceptions import ConfigError


@dataclass
class VideoConfig:
    source: object = 0
    resize_width: int = 640
    save_output: bool = False
    output_path: str = "outputs/output.mp4"


@dataclass
class DetectionConfig:
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.4


@dataclass
class TrackingConfig:
    max_age: int = 15
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    save_tracks_json: bool = False
    tracks_json_path: str = "logs/tracks.json"


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Loads config.yaml into an AppConfig object. Falls back to defaults if missing."""
    if not os.path.exists(path):
        return AppConfig()

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in '{path}': {e}")

    return AppConfig(
        video=VideoConfig(**raw.get("video", {})),
        detection=DetectionConfig(**raw.get("detection", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )


def parse_cli_args() -> argparse.Namespace:
    """CLI arguments override config.yaml values when provided."""
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Tracking")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--source", type=str, default=None, help="0 for webcam, or path to video file")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (0-1)")
    parser.add_argument("--model", type=str, default=None, help="YOLO model path, e.g. yolov8n.pt")
    parser.add_argument("--save-output", action="store_true", help="Save annotated video to outputs/")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a display window")
    return parser.parse_args()


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source is not None:
        config.video.source = int(args.source) if args.source.isdigit() else args.source
    if args.conf is not None:
        config.detection.confidence_threshold = args.conf
    if args.model is not None:
        config.detection.model_path = args.model
    if args.save_output:
        config.video.save_output = True
    return config