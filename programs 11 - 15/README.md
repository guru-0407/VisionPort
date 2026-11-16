# VisionPort — Programs 11 to 15

These are five independent Python programs for Tasks 11–15 in the VisionPort assignment.

## Files
- `task11_detect_traffic_congestion.py` — vehicle-count congestion alerts
- `task12_track_individual_vehicles.py` — vehicle tracking IDs with ByteTrack
- `task13_estimate_vehicle_speed.py` — approximate speed from tracked movement
- `task14_detect_traffic_signal_status.py` — red/yellow/green color detection in a configurable signal ROI
- `task15_estimate_traffic_waiting_time.py` — estimated stopped time for tracked vehicles in a configurable waiting zone
- `visionport_utils.py` — shared input/model/video-output helpers used by the five programs

## Run in Ubuntu / WSL
From the VisionPort project root, activate the same virtual environment used for Programs 1–10:

```bash
source ~/venv/bin/activate
cd /mnt/d/Study/VisionPort
```

Install dependencies if needed:

```bash
python -m pip install ultralytics opencv-python numpy
```

Keep `traffic_video.mp4` in the VisionPort project root. Run one program at a time:

```bash
python "programs 11 - 15/task11_detect_traffic_congestion.py"
python "programs 11 - 15/task12_track_individual_vehicles.py"
python "programs 11 - 15/task13_estimate_vehicle_speed.py"
python "programs 11 - 15/task14_detect_traffic_signal_status.py"
python "programs 11 - 15/task15_estimate_traffic_waiting_time.py"
```

The first YOLO run may download `yolov8n.pt`. Outputs are saved in an `outputs/` folder in the current working directory.

## Important limitations / settings
- **Task 11:** `CONGESTION_THRESHOLD` is a simple demonstration threshold. Change it to suit the clip.
- **Task 13:** `METERS_PER_PIXEL` must be calibrated using a known real-world distance in the camera view. The default value is illustrative only; do not report it as a calibrated speed.
- **Task 14:** the video must show a traffic signal. Set `SIGNAL_ROI = (x1, y1, x2, y2)` to cover the signal. The default ROI is the upper-right region and may need adjustment.
- **Task 15:** set `WAITING_ZONE = (x1, y1, x2, y2)` to the queue/stop-line region. A clip without vehicles stopping there may produce zero or very small waiting times.

These are educational demonstrations using a general YOLO model and simple image-processing rules, not certified traffic-control measurements.
