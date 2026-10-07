import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026")

IMAGE_DIR = DATA_DIR / "train" / "train_images"

ANNOTATION_FILE = (
    DATA_DIR
    / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)


# -----------------------------
# Load annotation file
# -----------------------------

with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)


# -----------------------------
# Select first image
# -----------------------------

image_info = data["images"][0]

image_id = image_info["id"]
filename = image_info["file_name"]

image_path = IMAGE_DIR / filename

print("Image:", filename)
print("Image ID:", image_id)


# -----------------------------
# Load image
# -----------------------------

image = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not find image: {image_path}"
    )

print("Image shape:", image.shape)


# -----------------------------
# Find annotations
# -----------------------------

annotations = [
    ann
    for ann in data["annotations"]
    if ann["image_id"] == image_id
]

print("Number of filaments:", len(annotations))


# -----------------------------
# Display image
# -----------------------------

plt.figure(figsize=(10, 10))

plt.imshow(
    image,
    cmap="gray"
)


# -----------------------------
# Draw masks
# -----------------------------

for annotation in annotations:

    segmentation = annotation["segmentation"]

    # Convert polygon → RLE
    rles = mask_utils.frPyObjects(
        segmentation,
        image.shape[0],
        image.shape[1]
    )

    rle = mask_utils.merge(rles)

    # Convert RLE → binary mask
    mask = mask_utils.decode(rle)

    plt.imshow(
        mask,
        alpha=0.4
    )


plt.title("Solar Filament Annotations")

plt.axis("off")


# -----------------------------
# Save instead of displaying
# -----------------------------

output_file = "filament_visualization.png"

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print(f"Saved visualization to {output_file}")