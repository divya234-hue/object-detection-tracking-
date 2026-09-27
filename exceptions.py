class VideoSourceError(Exception):
    """Raised when webcam/video file cannot be opened or read."""
    pass


class ModelLoadError(Exception):
    """Raised when the YOLO model fails to load."""
    pass


class DetectionError(Exception):
    """Raised when object detection fails on a given frame."""
    pass


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass