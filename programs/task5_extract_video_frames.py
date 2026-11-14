"""
PROGRAM 5 - Extract individual frames from a traffic video.

Saves one frame every FRAME_INTERVAL frames in the extracted_frames folder.
Change VIDEO_PATH if your video has a different name.
"""
import cv2
from pathlib import Path

VIDEO_PATH = "traffic_video.mp4"
FRAME_INTERVAL = 30
OUTPUT_DIR = Path("extracted_frames")
OUTPUT_DIR.mkdir(exist_ok=True)

video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    raise SystemExit(f"Error: Could not open '{VIDEO_PATH}'.")

frame_number = 0
saved_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    if frame_number % FRAME_INTERVAL == 0:
        filename = OUTPUT_DIR / f"frame_{frame_number:05d}.jpg"
        if cv2.imwrite(str(filename), frame):
            saved_count += 1
            print("Saved:", filename)

    frame_number += 1

video.release()
print(f"Program 5 completed. Read {frame_number} frames and saved {saved_count} images.")
