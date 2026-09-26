"""Train a YOLO detector on the reading-display labels.

The crop app exports labels into yolo_dataset/ (via the "Export YOLO dataset"
button) after you've corrected boxes on enough photos. Then train:

    .venv/bin/python train_yolo.py

The default checkpoint is an OBB model (yolov8n-obb.pt) matching the rotated
labels the app writes. Best weights land in runs/obb/train/weights/best.pt,
which the crop app picks up automatically on its next restart (YOLO OBB
inference replaces the heuristic). Recommended: label 200-400 diverse photos
(clear + dim + small meters), correct every box (including tilt) by hand
before exporting.
"""
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description="Train YOLO reading-detector")
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n-obb.pt",
                    help="backbone: yolov8n-obb.pt (fast) ... yolov8m-obb.pt, "
                         "yolo11n-obb.pt, or yolov8n.pt for plain detection")
    args = ap.parse_args()

    if not Path(args.data).is_file():
        print("No dataset yet:", args.data)
        print("Crop some images in the app and press 'Export YOLO dataset' first.")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,                 # GPU (RTX 4050)
        patience=30,
        cache=True,
        workers=4,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\nDone. Best weights:")
    print(f"  {best}")


if __name__ == "__main__":
    raise SystemExit(main())
