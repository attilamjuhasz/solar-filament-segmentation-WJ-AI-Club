import os
import json

# Define your data path based on the folder we saw earlier
DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

# Let's see what files or subfolders are inside the train directory
if os.path.exists(train_dir):
    print("Contents of train directory:", os.listdir(train_dir))