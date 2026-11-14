
"""
PROGRAM 7 - Detect bikes, buses and trucks using YOLO.
Saves the annotated output video.
"""

import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic_video.mp4"
OUTPUT_PATH = "multiple_vehicles_output.mp4"

# YOLO COCO class IDs
VEHICLE_CLASSES = [3, 5, 7]

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise SystemExit("Could not open traffic video.")

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

vehicle_counts = {
    "Motorcycles": 0,
    "Buses": 0,
    "Trucks": 0
}

print("Starting vehicle detection...")
print("Processing video. Please wait...")

while True:
    success, frame = video.read()

    if not success:
        break

    results = model.predict(
        frame,
        classes=VEHICLE_CLASSES,
        conf=0.25,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())

            if class_id == 3:
                vehicle_counts["Motorcycles"] += 1
            elif class_id == 5:
                vehicle_counts["Buses"] += 1
            elif class_id == 7:
                vehicle_counts["Trucks"] += 1

    writer.write(annotated_frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

video.release()
writer.release()

print("\nProgram 7 completed successfully!")
print("Frames processed:", frame_count)
print("Motorcycle detections:", vehicle_counts["Motorcycles"])
print("Bus detections:", vehicle_counts["Buses"])
print("Truck detections:", vehicle_counts["Trucks"])
print("Output saved as:", OUTPUT_PATH)