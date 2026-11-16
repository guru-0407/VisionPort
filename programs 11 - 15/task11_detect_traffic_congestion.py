"""Task 11: Detect traffic congestion and generate alerts."""
import cv2
from visionport_utils import load_model, open_video, make_writer, VEHICLE_CLASSES

# Demonstration rule: congestion is flagged when this many vehicles are visible.
CONGESTION_THRESHOLD = 8


def main():
    model = load_model()
    cap, fps, width, height, source = open_video()
    writer, output_path = make_writer("task11_congestion_alerts.mp4", fps, width, height)
    frame_no = 0
    alert_frames = 0
    print(f"Input: {source}")cd /mnt/d/Study/VisionPort
source ~/venv/bin/activate
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model(frame, verbose=False)[0]
        count = 0
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, VEHICLE_CLASSES[cls], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 0), 1)
        congested = count >= CONGESTION_THRESHOLD
        if congested:
            alert_frames += 1
        status = f"CONGESTION ALERT: {count} vehicles" if congested else f"Traffic normal: {count} vehicles"
        color = (0, 0, 255) if congested else (0, 200, 0)
        cv2.rectangle(frame, (10, 10), (min(width - 10, 510), 55), (25, 25, 25), -1)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
        writer.write(frame)
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames; current vehicle count: {count}")
    cap.release(); writer.release()
    print("\nTask 11 completed")
    print(f"Frames processed: {frame_no}")
    print(f"Frames with congestion alerts: {alert_frames}")
    print(f"Output saved: {output_path}")
    print(f"Rule used: congestion when vehicle count >= {CONGESTION_THRESHOLD}")

if __name__ == "__main__":
    main()
