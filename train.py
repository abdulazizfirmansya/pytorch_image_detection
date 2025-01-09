from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.info()
results = model.train(data='your_path', epochs = 20, imgsz=320)
