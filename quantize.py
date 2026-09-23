from ultralytics import YOLO

models = [YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v8/yolov8l.pt"),YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v26/yolo26l.pt"),
          YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v11/yolo11s.pt"),YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v26/yolo26n.pt"),
          YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v11/yolo11n.pt"),YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v26/yolo26s.pt"),
          YOLO("/Users/ayush/Desktop/model_compression/src/models/yolo/v11/yolo11l.pt")]
for model in models:
    model.export(format="litert", int8=True, data="coco128.yaml")