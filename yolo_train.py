from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="Dataset_v1/Dataset/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,
    lr0=0.01,
    workers=4,
    pretrained=True,
)
