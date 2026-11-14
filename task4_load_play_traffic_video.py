"""
PROGRAM 4 - Load and play a traffic video using OpenCV.

Change VIDEO_PATH to your video filename.
Press Q to stop playback.
"""
import cv2

VIDEO_PATH = "traffic_video.mp4"

video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    raise SystemExit(
        f"Error: Could not open '{VIDEO_PATH}'. "
        "Put a video in this folder or change VIDEO_PATH."
    )

fps = video.get(cv2.CAP_PROP_FPS)
delay = max(1, int(1000 / fps)) if fps and fps > 0 else 30

print("Playing video. Press Q to quit.")

while True:
    success, frame = video.read()
    if not success:
        print("Video playback finished.")
        break

    cv2.imshow("Program 4 - Traffic Video Playback", frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
