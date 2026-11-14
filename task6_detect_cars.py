
"""
PROGRAM 6 - Detect cars in a traffic video using YOLO.
The annotated video is saved instead of displayed in WSL.
"""

import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic_video.mp4"
OUTPUT_PATH = "detected_cars_output.mp4"

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise SystemExit(f"Error: Could not open {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    video.release()
    raise SystemExit("Could not create output video.")

frame_count = 0
total_cars = 0

print("Starting car detection using YOLO...")
print("Processing video. Please wait...")

while True:
    success, frame = video.read()

    if not success:
        break

    # Class ID 2 represents cars in COCO
    results = model.predict(
        frame,
        classes=[2],
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    car_count = len(results[0].boxes)
    total_cars += car_count
    frame_count += 1

    writer.write(annotated_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

video.release()
writer.release()

print("\nProgram 6 completed successfully!")
print("Frames processed:", frame_count)
print("Total car detections:", total_cars)
print("Output saved as:", OUTPUT_PATH)