import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils


DATA_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026")

IMAGE_DIR = DATA_DIR / "train" / "train_images"

ANNOTATION_FILE = (
    DATA_DIR / "train" /
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

OUTPUT_DIR = Path("data/category_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load annotations
with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)


# Pick one image
image_info = data["images"][0]

image_id = image_info["id"]
filename = image_info["file_name"]

image_path = IMAGE_DIR / filename

print("Image:", filename)
print("Image ID:", image_id)


# Load image
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(image_path)


height, width = image.shape


# Find annotations belonging to this image
annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] == image_id
]


# Create one mask for each category
category_masks = {}

for category_id in range(1, 5):

    mask = np.zeros((height, width), dtype=np.uint8)

    category_annotations = [
        ann for ann in annotations
        if ann["category_id"] == category_id
    ]

    for ann in category_annotations:

        segmentation = ann["segmentation"]

        rles = mask_utils.frPyObjects(
            segmentation,
            height,
            width
        )

        rle = mask_utils.merge(rles)

        decoded = mask_utils.decode(rle)

        mask = np.maximum(mask, decoded)

    category_masks[category_id] = mask


# Category names
category_names = {
    1: "Left",
    2: "Right",
    3: "Unidentifiable",
    4: "Ambiguous"
}


# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")


for index, category_id in enumerate(range(1, 5)):

    row = (index + 1) // 3
    col = (index + 1) % 3

    axes[row, col].imshow(image, cmap="gray")

    mask = category_masks[category_id]

    axes[row, col].imshow(
        mask,
        alpha=0.5
    )

    axes[row, col].set_title(
        f"{category_id}: {category_names[category_id]}"
    )

    axes[row, col].axis("off")


# Hide unused subplot
axes[1, 2].axis("off")


plt.tight_layout()

output_path = OUTPUT_DIR / "categories_example.png"

plt.savefig(output_path, dpi=150)

plt.close()

print()
print("Saved visualization to:")
print(output_path)
