import json

from dataset import SolarFilamentDataset
from torch.utils.data import Subset


# Main dataset folder
DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"

# Train/validation split
SPLIT_FILE = "data/train_val_split.json"


# Load the full dataset
dataset = SolarFilamentDataset(DATA_DIR)

print("Full dataset:", len(dataset))


# Load our split
with open(SPLIT_FILE, "r") as f:
    split = json.load(f)


training_ids = set(split["training"])
validation_ids = set(split["validation"])


# Find dataset indices for each split
training_indices = []
validation_indices = []

for index in range(len(dataset)):

    sample = dataset[index]

    # Your dataset returns a dictionary
    image_id = sample["image_id"]

    if image_id in training_ids:
        training_indices.append(index)

    elif image_id in validation_ids:
        validation_indices.append(index)


# Create train/validation datasets
train_dataset = Subset(dataset, training_indices)
val_dataset = Subset(dataset, validation_indices)


print()
print("Training dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))


# Test training sample
train_sample = train_dataset[0]

print()
print("===== TRAINING SAMPLE =====")
print("Image:", train_sample["filename"])
print("Image ID:", train_sample["image_id"])
print("Image shape:", train_sample["image"].shape)
print("Masks shape:", train_sample["masks"].shape)
print("Categories:", train_sample["categories"])


# Test validation sample
val_sample = val_dataset[0]

print()
print("===== VALIDATION SAMPLE =====")
print("Image:", val_sample["filename"])
print("Image ID:", val_sample["image_id"])
print("Image shape:", val_sample["image"].shape)
print("Masks shape:", val_sample["masks"].shape)
print("Categories:", val_sample["categories"])
