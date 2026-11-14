"""
PROGRAM 2 - Capture an image of a traffic signal using a camera.

Press S to save the camera image.
Press Q to quit.
"""
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Error: Could not open the camera. Check camera permission.")

print("Press S to capture and save an image. Press Q to quit.")

while True:
    success, frame = camera.read()
    if not success:
        print("Error: Could not read from camera.")
        break

    cv2.imshow("Program 2 - Traffic Image Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = "traffic_capture.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
