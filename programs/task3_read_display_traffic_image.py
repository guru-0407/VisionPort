"""
PROGRAM 3 - Read and display a traffic image using OpenCV.

Change IMAGE_PATH to the path of your traffic image.
Press any key in the image window to close it.
"""
import cv2

IMAGE_PATH = "traffic_capture.jpg"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise SystemExit(
        f"Error: Could not read '{IMAGE_PATH}'. "
        "Put a traffic image in this folder or change IMAGE_PATH."
    )

print("Image loaded successfully.")
print("Image dimensions (height, width, channels):", image.shape)

cv2.imshow("Program 3 - Traffic Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
