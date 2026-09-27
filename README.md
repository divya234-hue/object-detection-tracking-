# Object Detection and Tracking

## 1. Project Overview
A local, real-time Object Detection and Tracking application. It reads
video from a webcam or a video file, detects objects using a pre-trained
YOLOv8 model, and tracks each object across frames using the SORT
algorithm, assigning each one a consistent tracking ID.

## 2. Task Objective
Internship Task 4 — build a beginner-friendly YOLO + SORT pipeline that
runs fully locally, with no external APIs or API keys.

## 3. Features
- Real-time webcam or video file input (OpenCV)
- Pre-trained YOLOv8n object detection (80 COCO classes)
- Frame-by-frame bounding box + confidence filtering
- SORT-based multi-object tracking with stable IDs
- On-screen class name, tracking ID, and confidence per object
- FPS counter
- Keyboard controls (quit / save frame)
- Configurable confidence threshold and frame resizing for performance

## 4. Technologies Used
- Python
- OpenCV (video I/O, drawing, display)
- Ultralytics YOLOv8 (object detection)
- NumPy (array math)
- SciPy (Hungarian algorithm for matching)
- FilterPy (Kalman filter for motion prediction)

## 5. What is YOLO?
YOLO ("You Only Look Once") is an object detection neural network that
looks at the entire image in a single pass and predicts all bounding
boxes, class labels, and confidence scores at once. This makes it fast
enough for real-time video.

## 6. What is SORT?
SORT ("Simple Online and Realtime Tracking") takes the boxes YOLO finds
in each frame and figures out which box in this frame corresponds to
which box in the previous frame, using motion prediction (Kalman Filter)
and box overlap (IoU + Hungarian algorithm). This is what gives each
object a consistent ID as it moves.

## 7. Object Detection vs Object Tracking
- **Detection** (YOLO): "What objects are in this single frame, and where?"
  It has no memory of previous frames.
- **Tracking** (SORT): "Is this object the same one I saw in the last
  frame?" It links detections across frames and gives them a persistent ID.

Detection + Tracking together = YOLO finds objects every frame, SORT
connects them across frames.

## 8. Project Structure
object-detection-tracking/
│
├── main.py # entry point, video loop
├── detector.py # YOLO wrapper
├── tracker.py # SORT wrapper
├── utils.py # drawing helpers
├── requirements.txt
├── README.md
├── .gitignore
│
├── sort/
│ ├── init.py
│ └── sort.py # SORT algorithm implementation
│
└── models/ # YOLO weights auto-download here (empty by default)


## 9. Installation
```powershell
mkdir object-detection-tracking
cd object-detection-tracking

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 10. Running the Project
```powershell
python main.py
```
The first run will automatically download `yolov8n.pt` (~6MB).

## 11. Webcam Usage
Default. In `main.py`:
```python
VIDEO_SOURCE = 0
```

## 12. Video File Usage
```python
VIDEO_SOURCE = "video.mp4"
```
Place the video file in the project root, or give its full path.

## 13. Keyboard Controls
- `q` — quit the application
- `s` — save the current frame to `saved_frames/`

## 14. Example Output
Each detected object is drawn with:
Person | ID:1 | 0.87

Box color stays consistent per object class; tracking ID stays consistent
per physical object as long as it stays in frame.

## 15. Troubleshooting
| Problem | Fix |
|---|---|
| "Could not open webcam" | Check webcam is connected and not used by another app (Zoom, Teams, etc.) |
| "Video file not found" | Check the path/filename in `VIDEO_SOURCE` |
| Slow / laggy video | Lower `RESIZE_WIDTH`, or raise `CONFIDENCE_THRESHOLD` |
| Model download fails | Check internet connection; weights download on first run only |
| IDs keep changing rapidly | Increase `iou_threshold` slightly, or check for camera shake / motion blur |

## 16. Future Improvements
- Swap YOLOv8n for a larger model (`yolov8s`/`yolov8m`) if GPU available
- Add object counting / zone-crossing logic
- Log tracked object paths to a file
- Add a simple GUI for source/threshold selection

## Architecture

Video/Webcam
↓
OpenCV
↓
YOLO
↓
Object Detection
↓
SORT
↓
Object Tracking
↓
Bounding Boxes + Labels + IDs
↓
Real-time Display