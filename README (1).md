# Real-Time Object Detection — YOLOv8
### Detects 80 object classes from webcam, images, or video files

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```
> On first run, YOLOv8 automatically downloads the pre-trained weights (~6 MB for nano model).

---

### Step 2 — Run

| Mode | Command |
|---|---|
| Webcam (default) | `python yolo_detector.py` |
| Custom webcam index | `python yolo_detector.py --cam 1` |
| Image file | `python yolo_detector.py --source photo.jpg` |
| Video file | `python yolo_detector.py --source video.mp4` |

---

## Model Options

| Flag | Model | Speed | Accuracy | Size |
|---|---|---|---|---|
| `--model yolov8n.pt` | Nano | Fastest | Good | ~6 MB |
| `--model yolov8s.pt` | Small | Fast | Better | ~22 MB |
| `--model yolov8m.pt` | Medium | Moderate | Great | ~52 MB |
| `--model yolov8l.pt` | Large | Slow | Very good | ~87 MB |
| `--model yolov8x.pt` | XLarge | Slowest | Best | ~136 MB |

---

## All Flags

```
--source     webcam | image.jpg | video.mp4   (default: webcam)
--model      yolov8n/s/m/l/x.pt               (default: yolov8n.pt)
--conf       0.0 – 1.0  confidence threshold   (default: 0.40)
--iou        0.0 – 1.0  NMS IoU threshold      (default: 0.45)
--cam        webcam device index               (default: 0)
--no-labels  hide class name labels
--no-conf    hide confidence scores
--no-fps     hide FPS/HUD overlay
```

---

## Examples

```bash
# High accuracy mode on webcam
python yolo_detector.py --model yolov8m.pt --conf 0.5

# Run on an image, hide confidence scores
python yolo_detector.py --source street.jpg --no-conf

# Run on a video file with lower threshold to catch more objects
python yolo_detector.py --source traffic.mp4 --conf 0.3

# Use second webcam with large model
python yolo_detector.py --cam 1 --model yolov8l.pt
```

---

## Output

- **Webcam / Video**: live window with bounding boxes, labels, FPS counter
- **Image**: annotated image saved as `<filename>_detected.<ext>` in same folder
- **Video**: annotated video saved as `<filename>_detected.mp4` in same folder

---

## Controls

| Key | Action |
|---|---|
| `Q` | Quit / close window |
| Any key | Close image result window |

---

## What it detects (COCO 80 classes)

People, vehicles (car, bus, truck, bicycle, motorcycle, airplane, boat),
animals (dog, cat, bird, horse, cow, elephant, bear, zebra, giraffe),
everyday objects (chair, bottle, laptop, phone, book, clock, cup, fork),
and many more.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ultralytics` | Run `pip install ultralytics` |
| Webcam not opening | Try `--cam 1` or `--cam 2` |
| Very low FPS | Switch to `--model yolov8n.pt` (nano) |
| Too many false detections | Increase `--conf 0.6` |
| Missing small objects | Decrease `--conf 0.25` |
