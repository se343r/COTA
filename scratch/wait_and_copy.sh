#!/bin/bash
while pgrep -f "train_yolo_electronic.py" > /dev/null; do
    sleep 5
done
cp runs/detect/runs/detect/yolov8n_electronic_bbox/weights/best.pt yolov8n-electronic-bbox.pt
