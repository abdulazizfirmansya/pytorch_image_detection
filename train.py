from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.info()

results = model.train(data='D:\Branch_counter\data\custom_data2.yaml', epochs = 20, imgsz=320)