
from ultralytics import YOLO
import torch

assert torch.cuda.is_available(), "Enable a GPU in Kaggle Settings"

student = YOLO("yolo26n.pt")

results = student.train(
    data="/kaggle/working/coco_minitrain_10k.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    distill_model="yolo26x.pt",
    project="/kaggle/working/runs",
    name="yolo26n_distilled",
    exist_ok=True,
    patience=20,
    plots=True,
)

print("Best model:", "/kaggle/working/runs/yolo26n_distilled/weights/best.pt")
