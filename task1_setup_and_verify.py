"""
PROGRAM 1 - Install and verify Python, OpenCV and required libraries.

Run the installation commands in the terminal first:
    python3 -m pip install opencv-python numpy ultralytics

Then run:
    python3 Program1.py
"""
import sys

print("VisionPort - Program 1: Environment Verification")
print("Python version:", sys.version)

packages = {
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "ultralytics": "Ultralytics / YOLO"
}

for module_name, display_name in packages.items():
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "Version not available")
        print(f"[OK] {display_name}: {version}")
    except ImportError:
        print(f"[MISSING] {display_name}. Install it using:")
        print("    python3 -m pip install opencv-python numpy ultralytics")

print("\\nProgram 1 completed.")