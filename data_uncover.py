import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 1. Scan the current workspace directory to find any zip files
current_dir = os.getcwd()
print(f"Scanning directory: {current_dir}")
all_files = os.listdir(current_dir)

# Look for any zip file containing 'filament' or matching your dataset
zip_name = None
for file in all_files:
    if "filament" in file.lower() and file.endswith(".zip"):
        zip_name = file
        break

if not zip_name:
    print(f"❌ Error: Could not find any dataset .zip file in {current_dir}!")
    print(f"Available files are: {all_files}")
    print("Please make sure your Kaggle dataset zip file is uploaded here.")
    exit()

extract_dir = "filament_dataset"

# 2. Extract the found dataset zip
if not os.path.exists(extract_dir):
    print(f"📦 Found dataset zip: '{zip_name}'. Extracting to '{extract_dir}'...")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
else:
    print(f"📂 Dataset folder '{extract_dir}' already exists. Skipping extraction.")

# 3. Print the structure inside the unzipped folder so we know exactly where images are
print("\n--- Unzipped Folder Contents ---")


def print_tree(dir_path, level=0):
    if not os.path.exists(dir_path):
        return
    for item in os.listdir(dir_path)[:5]:  # Show up to 5 items per folder
        full_path = os.path.join(dir_path, item)
        print("  " * level + f"|- {item}")
        if os.path.isdir(full_path) and level < 2:
            print_tree(full_path, level + 1)


print_tree(extract_dir)
print("--------------------------------\n")


# 4. Pure TF function to safely read the H-Alpha images
def load_images_with_tf(actual_path, target_size=(256, 256)):
    if not os.path.exists(actual_path):
        return np.array()

    images = []
    file_names = sorted(
        [
            f
            for f in os.listdir(actual_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    for img_name in file_names:
        img_path = os.path.join(actual_path, img_name)
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_image(
            img_raw, channels=1, expand_animations=False
        )
        img = tf.image.resize(img, target_size)
        images.append(img.numpy() / 255.0)

    return np.array(images, dtype=np.float32)


# Try to resolve paths dynamically based on common Kaggle paths
base_path = extract_dir
if "MAGFiLO_1.0_Kaggle_2026" in os.listdir(extract_dir):
    base_path = os.path.join(extract_dir, "MAGFiLO_1.0_Kaggle_2026")

train_img_dir = os.path.join(base_path, "train", "train_images")
test_img_dir = os.path.join(base_path, "test", "test_images")

# Fallback check if folders are named flatly
if not os.path.exists(train_img_dir):
    train_img_dir = os.path.join(base_path, "train_images")
    test_img_dir = os.path.join(base_path, "test_images")

X_train = load_images_with_tf(train_img_dir)
X_test = load_images_with_tf(test_img_dir)

print(f"Final X_train array shape: {X_train.shape}")
print(f"Final X_test array shape: {X_test.shape}")
