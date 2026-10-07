import json
import random
from pathlib import Path


# -----------------------------
# Settings
# -----------------------------

DATA_DIR = Path(
    "data/MAGFiLO_1.0_Kaggle_2026"
)

ANNOTATION_FILE = (
    DATA_DIR
    / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# -----------------------------
# Load JSON
# -----------------------------

with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)


images = data["images"]

num_images = len(images)

print("Total images:", num_images)


# -----------------------------
# Create shuffled indices
# -----------------------------

indices = list(range(num_images))

random.seed(RANDOM_SEED)

random.shuffle(indices)


# -----------------------------
# Split
# -----------------------------

validation_size = int(
    num_images * VALIDATION_RATIO
)

validation_indices = indices[
    :validation_size
]

training_indices = indices[
    validation_size:
]


# -----------------------------
# Convert indices to image IDs
# -----------------------------

training_ids = [
    images[i]["id"]
    for i in training_indices
]

validation_ids = [
    images[i]["id"]
    for i in validation_indices
]


# -----------------------------
# Save split
# -----------------------------

split = {
    "training": training_ids,
    "validation": validation_ids
}


with open("data/train_val_split.json", "w") as f:
    json.dump(
        split,
        f,
        indent=2
    )


# -----------------------------
# Print results
# -----------------------------

print()
print("Training images:", len(training_ids))
print("Validation images:", len(validation_ids))

print()
print("Saved to:")
print("data/train_val_split.json")

print()
print("First 5 training images:")

for image_id in training_ids[:5]:
    print(image_id)

print()
print("First 5 validation images:")

for image_id in validation_ids[:5]:
    print(image_id)