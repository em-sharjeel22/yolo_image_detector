"""
Real-Time Object Detection using YOLOv8
========================================
Uses Ultralytics YOLOv8 (pre-trained on COCO — 80 classes).
Supports: webcam live feed | image file | video file
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF logs if TF is installed

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "model"      : "yolov8m.pt",   # nano=fastest | s/m/l/x=more accurate
    "confidence" : 0.50,           # minimum confidence threshold (0–1)
    "iou"        : 0.55,           # IoU threshold for NMS
    "img_size"   : 640,            # inference image size
    "device"     : "",             # "" = auto (GPU if available, else CPU)
    "show_fps"   : True,
    "show_labels": True,
    "show_conf"  : True,
}

# 20 visually distinct BGR colours for class labels
PALETTE = [
    (56,  182, 255), (0,   200, 83),  (255, 87,  34),  (156, 39,  176),
    (3,   169, 244), (255, 193, 7),   (233, 30,  99),  (0,   188, 212),
    (139, 195, 74),  (255, 152, 0),   (96,  125, 139), (244, 67,  54),
    (63,  81,  181), (0,   150, 136), (205, 220, 57),  (121, 85,  72),
    (255, 235, 59),  (33,  150, 243), (76,  175, 80),  (103, 58,  183),
]

def get_colour(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_box(frame, box, class_id, class_name, confidence, show_conf=True):
    x1, y1, x2, y2 = map(int, box)
    colour = get_colour(class_id)

    # Bounding box (2px)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label text
    label = f"{class_name} {confidence:.2f}" if show_conf else class_name
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    # Filled label background
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), colour, -1)

    # White label text
    cv2.putText(frame, label, (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps, num_detections, model_name, source_label):
    """Draw top-left HUD with FPS, detection count, and model info."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS : {fps:5.1f}",          (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)
    cv2.putText(frame, f"Objects : {num_detections}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(frame, f"Model : {model_name}",       (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Source label (top-right)
    label = f"Source: {source_label}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ─── Core inference ───────────────────────────────────────────────────────────

def process_frame(model, frame, cfg):
    results = model(
        frame,
        conf     = cfg["confidence"],
        iou      = cfg["iou"],
        imgsz    = cfg["img_size"],
        device   = cfg["device"],
        verbose  = False,
    )[0]

    detections = 0
    if results.boxes is not None:
        for box in results.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            coords     = box.xyxy[0].tolist()

            if cfg["show_labels"]:
                draw_box(frame, coords, class_id, class_name, confidence, cfg["show_conf"])
            detections += 1

    return frame, detections


# ─── Mode: Webcam (real-time) ─────────────────────────────────────────────────

def run_webcam(model, cfg, cam_index=0):
    print(f"\n[INFO] Opening webcam (index {cam_index}) ... press Q to quit.\n")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check the camera index or connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps       = 0.0
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame received — retrying ...")
            continue

        frame, num_det = process_frame(model, frame, cfg)

        # FPS calculation (rolling average over 10 frames)
        frame_count += 1
        if frame_count % 10 == 0:
            now  = time.time()
            fps  = 10.0 / (now - prev_time)
            prev_time = now

        if cfg["show_fps"]:
            draw_hud(frame, fps, num_det, cfg["model"], f"Webcam {cam_index}")

        cv2.imshow("YOLO Real-Time Detection  [Q = quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam session ended.")


# ─── Mode: Image file ─────────────────────────────────────────────────────────

def run_image(model, cfg, image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    print(f"\n[INFO] Running detection on image: {path.name}")
    frame = cv2.imread(str(path))
    if frame is None:
        print("[ERROR] Failed to read the image file.")
        return

    t0 = time.perf_counter()
    frame, num_det = process_frame(model, frame, cfg)
    elapsed = time.perf_counter() - t0

    fps = 1.0 / elapsed if elapsed > 0 else 0
    draw_hud(frame, fps, num_det, cfg["model"], path.name)

    print(f"[INFO] Detected {num_det} object(s) in {elapsed*1000:.1f} ms")

    # Save result
    out_path = path.parent / f"{path.stem}_detected{path.suffix}"
    cv2.imwrite(str(out_path), frame)
    print(f"[INFO] Saved result → {out_path}")

    cv2.imshow(f"YOLO Detection — {path.name}  [any key = close]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Mode: Video file ─────────────────────────────────────────────────────────

def run_video(model, cfg, video_path: str):
    path = Path(video_path)
    if not path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    print(f"\n[INFO] Running detection on video: {path.name}  [Q = quit]")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        return

    # Writer for saving output
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = path.parent / f"{path.stem}_detected.mp4"
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    fps         = 0.0
    prev_time   = time.time()
    frame_count = 0
    total_det   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, num_det = process_frame(model, frame, cfg)
        total_det += num_det
        frame_count += 1

        if frame_count % 10 == 0:
            now  = time.time()
            fps  = 10.0 / (now - prev_time)
            prev_time = now

        draw_hud(frame, fps, num_det, cfg["model"], path.name)
        writer.write(frame)
        cv2.imshow(f"YOLO Detection — {path.name}  [Q = quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_det = total_det / max(frame_count, 1)
    print(f"[INFO] Processed {frame_count} frames | avg {avg_det:.1f} detections/frame")
    print(f"[INFO] Saved result → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detector")
    parser.add_argument("--source",     default="webcam",     help="webcam | path/to/image.jpg | path/to/video.mp4")
    parser.add_argument("--model",      default="yolov8m.pt", help="yolov8m/s/m/l/x.pt")
    parser.add_argument("--conf",       type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--iou",        type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--cam",        type=int,   default=0,    help="Webcam index")
    parser.add_argument("--no-labels",  action="store_true",      help="Hide class labels")
    parser.add_argument("--no-conf",    action="store_true",      help="Hide confidence scores")
    parser.add_argument("--no-fps",     action="store_true",      help="Hide FPS overlay")
    args = parser.parse_args()

    # Override CONFIG with CLI args
    CONFIG["model"]       = args.model
    CONFIG["confidence"]  = args.conf
    CONFIG["iou"]         = args.iou
    CONFIG["show_labels"] = not args.no_labels
    CONFIG["show_conf"]   = not args.no_conf
    CONFIG["show_fps"]    = not args.no_fps

    # Load model (downloads weights on first run)
    print(f"\n[INFO] Loading model: {CONFIG['model']} ...")
    model = YOLO(CONFIG["model"])
    model_name = Path(CONFIG["model"]).stem
    CONFIG["model"] = model_name

    print(f"[INFO] Model loaded — {len(model.names)} classes available")
    print(f"[INFO] Confidence threshold : {CONFIG['confidence']}")
    print(f"[INFO] IoU threshold        : {CONFIG['iou']}")

    # Route to correct mode
    src = args.source.lower()
    if src == "webcam":
        run_webcam(model, CONFIG, cam_index=args.cam)
    elif Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image(model, CONFIG, args.source)
    elif Path(args.source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        run_video(model, CONFIG, args.source)
    else:
        print(f"[ERROR] Unknown source: '{args.source}'")
        print("        Use: --source webcam | image.jpg | video.mp4")


if __name__ == "__main__":
    main()
