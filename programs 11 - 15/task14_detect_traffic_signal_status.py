"""Task 14: Detect red, yellow, or green traffic signal color in a configurable ROI.

Set SIGNAL_ROI to the traffic-light region in your video: (x1, y1, x2, y2).
The default ROI is the top-right quarter; change it to match your footage.
"""
import cv2
import numpy as np
from visionport_utils import open_video, make_writer

SIGNAL_ROI = None  # Example: (500, 0, 640, 180). None uses top-right quarter.
MIN_COLOR_PIXELS = 12


def classify_signal(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 90, 70]), np.array([180, 255, 255]))
    red = cv2.countNonZero(red1) + cv2.countNonZero(red2)
    yellow = cv2.countNonZero(cv2.inRange(hsv, np.array([15, 80, 80]), np.array([38, 255, 255])))
    green = cv2.countNonZero(cv2.inRange(hsv, np.array([40, 70, 60]), np.array([95, 255, 255])))
    counts = {"RED": red, "YELLOW": yellow, "GREEN": green}
    color, amount = max(counts.items(), key=lambda item: item[1])
    return (color if amount >= MIN_COLOR_PIXELS else "NO SIGNAL DETECTED"), counts


def main():
    cap, fps, width, height, source = open_video()
    writer, output_path = make_writer("task14_signal_status.mp4", fps, width, height)
    if SIGNAL_ROI is None:
        x1, y1, x2, y2 = int(width * .70), 0, width, int(height * .45)
    else:
        x1, y1, x2, y2 = SIGNAL_ROI
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("SIGNAL_ROI is invalid. Use (x1, y1, x2, y2) inside the video frame.")
    frame_no = 0
    status_counts = {"RED": 0, "YELLOW": 0, "GREEN": 0, "NO SIGNAL DETECTED": 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        roi = frame[y1:y2, x1:x2]
        status, _ = classify_signal(roi)
        status_counts[status] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}.get(status, (255, 255, 255))
        cv2.putText(frame, f"Signal: {status}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
        writer.write(frame)
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames; latest signal status: {status}")
    cap.release(); writer.release()
    print("\nTask 14 completed")
    print(f"Frames processed: {frame_no}")
    for key, value in status_counts.items():
        print(f"{key} frames: {value}")
    print(f"Output saved: {output_path}")
    print("If no signal was detected, adjust SIGNAL_ROI to cover the actual traffic light.")

if __name__ == "__main__":
    main()
