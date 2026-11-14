
"""
PROGRAM 10 - Traffic Density Detection Using YOLO.
Classifies traffic as Low, Medium or High.
Saves the annotated output video.
"""

import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic_video.mp4"
OUTPUT_PATH = "traffic_density_output.mp4"

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
density_counts = {
    "Low": 0,
    "Medium": 0,
    "High": 0
}

print("Starting traffic density analysis...")

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
        len(result.boxes)
        if result.boxes is not None else 0
    )

    # Classify traffic density
    if vehicle_count <= 4:
        density = "Low"
    elif vehicle_count <= 10:
        density = "Medium"
    else:
        density = "High"

    density_counts[density] += 1

    annotated = result.plot()

    cv2.putText(
        annotated,
        f"Vehicle Count: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Traffic Density: {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    writer.write(annotated)
    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Frame {frame_count}: "
            f"{vehicle_count} vehicles - {density} traffic"
        )

video.release()
writer.release()

print("\nProgram 10 completed successfully!")
print("Frames processed:", frame_count)
print("Low traffic frames:", density_counts["Low"])
print("Medium traffic frames:", density_counts["Medium"])
print("High traffic frames:", density_counts["High"])
print("Output saved as:", OUTPUT_PATH)
