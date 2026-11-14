# VisionPort - Deep Learning Assignment

This folder contains 10 independent Python programs for the first 10 VisionPort tasks.

## Files
- task1_setup_and_verify.py
- task2_capture_traffic_image.py
- task3_read_display_traffic_image.py
- task4_load_play_traffic_video.py
- task5_extract_video_frames.py
- task6_detect_cars.py
- task7_detect_multiple_vehicles.py
- task8_display_labels_confidence.py
- task9_count_vehicles.py
- task10_traffic_density.py

## Setup in Ubuntu / WSL
Open the terminal in this folder and run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run programs separately
Run one task at a time, for example:

```bash
python3 task1_setup_and_verify.py
```

Then run task 2:

```bash
python3 task2_capture_traffic_image.py
```

Continue with the corresponding file for each task. Do not run all programs at once.

## Input files
- Task 3 expects `traffic_capture.jpg` in this folder by default. Task 2 creates this file when you press S.
- Tasks 4-10 expect a video named `traffic_video.mp4` in this folder by default.
- If you use different filenames, edit IMAGE_PATH or VIDEO_PATH at the top of the relevant Python file.

## YOLO model
Tasks 6-10 use Ultralytics YOLOv8 Nano and may download `yolov8n.pt` on first run. Internet is required for the initial download.

## Traffic density thresholds in Task 10
- Low: 0-4 vehicles
- Medium: 5-10 vehicles
- High: 11 or more vehicles

These are simple demonstration thresholds for this student project, not calibrated traffic-engineering standards.
