"""Task 12: Track individual vehicles across frames using YOLO tracking IDs."""
import cv2
from visionport_utils import load_model, open_video, make_writer, VEHICLE_CLASSES


def main():
    model = load_model()
    cap, fps, width, height, source = open_video()
    writer, output_path = make_writer("task12_vehicle_tracking.mp4", fps, width, height)
    frame_no = 0
    seen_ids = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            boxes = result.boxes.xyxy.int().cpu().tolist()
            for track_id, cls, (x1, y1, x2, y2) in zip(ids, classes, boxes):
                if cls not in VEHICLE_CLASSES:
                    continue
                seen_ids.add(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
                cv2.putText(frame, f"{VEHICLE_CLASSES[cls]} ID:{track_id}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 160, 0), 2)
        cv2.putText(frame, f"Unique vehicle IDs: {len(seen_ids)}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        writer.write(frame)
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames; unique IDs so far: {len(seen_ids)}")
    cap.release(); writer.release()
    print("\nTask 12 completed")
    print(f"Frames processed: {frame_no}")
    print(f"Unique tracking IDs observed: {len(seen_ids)}")
    print(f"Output saved: {output_path}")

if __name__ == "__main__":
    main()
