import json
import os

# Path to the training annotation JSON file
json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

with open(json_path, "r") as f:
    coco_data = json.load(f)

# COCO datasets typically contain 'images', 'annotations', and 'categories' keys
print("Keys in JSON:", coco_data.keys())
print(f"Total images in training set: {len(coco_data.get('images', []))}")
print(f"Total annotations: {len(coco_data.get('annotations', []))}")
print(f"Categories: {coco_data.get('categories', [])}")

# Print info about the very first image to see what its metadata looks like
if coco_data.get('images'):
    print("\nSample image metadata:", coco_data['images'][0])