"""Task 15: Estimate how long tracked vehicles wait while stopped near a signal.

WAITING_ZONE should be set to the stop-line/queue area in the video: (x1,y1,x2,y2).
This is a simple video-based estimate, not an official traffic-engineering measurement.
"""
import cv2
from collections import defaultdict
from visionport_utils import load_model, open_video, make_writer, VEHICLE_CLASSES

WAITING_ZONE = None  # Example: (100, 250, 620, 470). None uses lower half of frame.
STOPPED_PIXEL_THRESHOLD = 3.0  # Max center movement between sampled frames.


def main():
    model = load_model()
    cap, fps, width, height, source = open_video()
    writer, output_path = make_writer("task15_waiting_time.mp4", fps, width, height)
    if WAITING_ZONE is None:
        x1, y1, x2, y2 = 0, height // 2, width, height
    else:
        x1, y1, x2, y2 = WAITING_ZONE
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("WAITING_ZONE is invalid. Use (x1, y1, x2, y2) inside the video frame.")
    state = defaultdict(lambda: {"last_center": None, "waiting_frames": 0, "total_wait_seconds": 0.0})
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        active_waiters = 0
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            for track_id, cls, box in zip(ids, classes, boxes):
                if cls not in VEHICLE_CLASSES:
                    continue
                bx1, by1, bx2, by2 = map(int, box)
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                in_zone = x1 <= cx <= x2 and y1 <= cy <= y2
                item = state[track_id]
                stopped = False
                if in_zone and item["last_center"] is not None:
                    px, py = item["last_center"]
                    movement = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    stopped = movement <= STOPPED_PIXEL_THRESHOLD
                if in_zone and stopped:
                    item["waiting_frames"] += 1
                    item["total_wait_seconds"] += 1.0 / fps
                    active_waiters += 1
                item["last_center"] = (cx, cy) if in_zone else None
                label = f"ID {track_id} wait {item['total_wait_seconds']:.1f}s" if in_zone else f"ID {track_id}"
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 180, 255) if stopped else (255, 150, 0), 2)
                cv2.putText(frame, label, (bx1, max(20, by1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"Waiting vehicles in zone: {active_waiters}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        writer.write(frame)
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames; waiting vehicles in zone: {active_waiters}")
    cap.release(); writer.release()
    wait_values = [v["total_wait_seconds"] for v in state.values() if v["total_wait_seconds"] > 0]
    print("\nTask 15 completed")
    print(f"Frames processed: {frame_no}")
    print(f"Vehicles with recorded waiting time: {len(wait_values)}")
    print(f"Total estimated waiting time: {sum(wait_values):.1f} seconds")
    if wait_values:
        print(f"Average waiting time per vehicle: {sum(wait_values) / len(wait_values):.1f} seconds")
    print(f"Output saved: {output_path}")
    print("Set WAITING_ZONE to the actual queue/stop-line area for a meaningful estimate.")

if __name__ == "__main__":
    main()
