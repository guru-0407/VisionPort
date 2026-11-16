"""Shared helpers for VisionPort Tasks 11-15."""
from pathlib import Path
import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def find_video():
    candidates = [Path("traffic_video.mp4"), Path("../traffic_video.mp4"), Path(__file__).resolve().parent.parent / "traffic_video.mp4"]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("traffic_video.mp4 was not found. Put it in the VisionPort project root folder.")

def load_model():
    return YOLO("yolov8n.pt")

def open_video():
    path = find_video()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, width, height, path

def make_writer(filename, fps, width, height):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / filename
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer, path

def is_vehicle(class_id):
    return int(class_id) in VEHICLE_CLASSES
