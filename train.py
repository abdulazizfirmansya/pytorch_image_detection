"""
this program how to train object detection using your own custom datasets
"""

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.info()
results = model.train(data='your_path', epochs = 20, imgsz=320)
# data: your path store the dataset
# epochs : iteration of your own learn
# imgsz = image size
