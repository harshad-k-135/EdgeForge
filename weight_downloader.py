from pathlib import Path
import shutil
from ultralytics import YOLO

models = {
	"yolov8s.pt": Path("models/yolo/v8"),
	"yolo11s.pt": Path("models/yolo/v11"),
	"yolo26s.pt": Path("models/yolo/v26"),
}

for model_name, model_dir in models.items():
	model_dir.mkdir(parents=True, exist_ok=True)

	YOLO(model_name)
	source = Path(model_name)
	destination = model_dir / model_name

	shutil.copy2(source, destination)
	print(f"Model downloaded to: {destination}")

