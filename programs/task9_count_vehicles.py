
"""
PROGRAM 9 - Count the number of vehicles in each video frame.
Using YOLOv8 and OpenCV.
"""

import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic_video.mp4"
OUTPUT_PATH = "vehicle_count_output.mp4"

# Car, motorcycle, bus and truck
VEHICLE_CLASSES = [2, 3, 5, 7]

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
total_vehicle_detections = 0

print("Starting vehicle counting...")

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

    vehicle_count = (
        len(result.boxes) if result.boxes is not None else 0
    )

    total_vehicle_detections += vehicle_count

    annotated = result.plot()

    cv2.putText(
        annotated,
        f"Vehicles in frame: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    writer.write(annotated)
    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Frame {frame_count}: "
            f"{vehicle_count} vehicles detected"
        )

video.release()
writer.release()

print("\nProgram 9 completed successfully!")
print("Frames processed:", frame_count)
print("Total vehicle detections across frames:", total_vehicle_detections)
print("Output saved as:", OUTPUT_PATH)
