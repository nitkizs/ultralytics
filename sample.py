# import os

# # Paths to your folders
# jpeg_folder = "/home/nitkizs/coco/images/val2017"
# txt_folder = "/home/nitkizs/projects/rt_detr/datasets/coco/images/val2017"

# # Count .jpeg files
# jpeg_count = len([f for f in os.listdir(jpeg_folder) if f.lower().endswith(".jpg")])
# print(f"Number of .jpeg files in '{jpeg_folder}': {jpeg_count}")

# # # Count .txt files
# # txt_count = len([f for f in os.listdir(txt_folder) if f.lower().endswith(".txt")])
# # print(f"Number of .txt files in '{txt_folder}': {txt_count}")
# # Count .jpeg files
# jpeg_count = len([f for f in os.listdir(jpeg_folder) if f.lower().endswith(".jpg")])
# print(f"Number of .jpg files in new coco datatset '{txt_folder}': {jpeg_count}")
from ultralytics import YOLO

# Load a model
model = YOLO("projects/rt_detr/rtdetr-l.pt")  # load a pretrained model (recommended for training)

# # Train the model with 2 GPUs
# results = model.train(data="projects/rt_detr/ultralytics/ultralytics/cfg/datasets/coco.yaml", epochs=100, imgsz=640, device=[0, 1])

# Train the model with the two most idle GPUs
results = model.train(data="/home/nitkizs/projects/rt_detr/ultralytics/ultralytics/cfg/datasets/coco.yaml", epochs=100, imgsz=640)