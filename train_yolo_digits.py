from ultralytics import YOLO
import shutil
from pathlib import Path

def main():
    model = YOLO("yolov8n.yaml") 
    
    results = model.train(
        data="yolo_digits_dataset/dataset.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="yolov8n_digits_bbox",
        exist_ok=True,
        patience=30
    )
    
    # Path to the best weights
    best_weights = Path("runs/detect/yolov8n_digits_bbox/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, "yolov8n-digits-bbox.pt")
        print("Training complete! Best weights saved to yolov8n-digits-bbox.pt")
    else:
        print("Could not find best.pt!")

if __name__ == "__main__":
    main()
