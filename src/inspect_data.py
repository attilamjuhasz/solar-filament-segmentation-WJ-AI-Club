import json
from pathlib import Path


DATA_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026")

ANNOTATION_FILE = (
    DATA_DIR
    / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)


with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)


print("===== DATASET =====")

print("Images:", len(data["images"]))
print("Annotations:", len(data["annotations"]))
print("Categories:", len(data["categories"]))


print("\n===== FIRST IMAGE =====")

image = data["images"][0]

print("ID:", image["id"])
print("Filename:", image["file_name"])
print("Width:", image["width"])
print("Height:", image["height"])


print("\n===== FIRST ANNOTATION =====")

annotation = data["annotations"][0]

print("Image ID:", annotation["image_id"])
print("Category:", annotation["category_id"])
print("Area:", annotation["area"])
print("Bounding box:", annotation["bbox"])

print("Segmentation type:", type(annotation["segmentation"]))


print("\n===== CATEGORIES =====")

for category in data["categories"]:
    print(category)