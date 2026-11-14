
"""
PROGRAM 8 - Display Vehicle Names and Confidence Scores
Using YOLOv8 and OpenCV.
"""

import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic_video.mp4"
OUTPUT_PATH = "vehicle_labels_output.mp4"

# Car, Motorcycle, Bus and Truck
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

print("Starting vehicle detection with confidence scores...")

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
    annotated = result.plot()

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            label = f"{name}: {confidence * 100:.1f}%"

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    writer.write(annotated)
    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

video.release()
writer.release()

print("\nProgram 8 completed successfully!")
print("Frames processed:", frame_count)
print("Output saved as:", OUTPUT_PATH)
