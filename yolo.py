from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())

    model = YOLO("yolov8n.pt")

    results = model.train(data="./data.yaml", epochs=20, imgsz=256, optimizer = 'Adam', lr0= 0.005)

    print('foi')
