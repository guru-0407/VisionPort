"""Task 13: Estimate vehicle speed from tracked movement.

IMPORTANT: Set METERS_PER_PIXEL for the camera view before treating speeds as meaningful.
The default is only a demonstration value and is not a calibrated measurement.
"""
import cv2
from collections import defaultdict, deque
from visionport_utils import load_model, open_video, make_writer, VEHICLE_CLASSES

METERS_PER_PIXEL = 0.05  # Replace after calibrating a known distance in the scene.


def main():
    model = load_model()
    cap, fps, width, height, source = open_video()
    writer, output_path = make_writer("task13_speed_estimation.mp4", fps, width, height)
    histories = defaultdict(lambda: deque(maxlen=2))
    frame_no = 0
    latest_speeds = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            for track_id, cls, box in zip(ids, classes, boxes):
                if cls not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                histories[track_id].append((frame_no, center))
                speed_text = "Speed: calculating"
                if len(histories[track_id]) == 2:
                    (f0, p0), (f1, p1) = histories[track_id]
                    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                    pixels = (dx * dx + dy * dy) ** 0.5
                    elapsed = (f1 - f0) / fps
                    speed_kmh = (pixels * METERS_PER_PIXEL / elapsed) * 3.6 if elapsed > 0 else 0.0
                    latest_speeds[track_id] = speed_kmh
                    speed_text = f"{speed_kmh:.1f} km/h*"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 200), 2)
                cv2.putText(frame, f"ID {track_id} {speed_text}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 200), 2)
        cv2.putText(frame, "*Approximate; calibrate METERS_PER_PIXEL", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        writer.write(frame)
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames")
    cap.release(); writer.release()
    print("\nTask 13 completed")
    print(f"Frames processed: {frame_no}")
    print(f"Vehicles with speed estimates: {len(latest_speeds)}")
    if latest_speeds:
        print(f"Average estimated speed: {sum(latest_speeds.values()) / len(latest_speeds):.1f} km/h")
    print(f"Output saved: {output_path}")
    print("NOTE: Speed values are demonstrations until METERS_PER_PIXEL is calibrated for the camera.")

if __name__ == "__main__":
    main()
