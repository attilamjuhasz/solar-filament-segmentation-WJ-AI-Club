import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import json
import cv2

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

base_path = os.path.join(extract_dir, "MAGFiLO_1.0_Kaggle_2026")
train_dir = os.path.join(base_path, "train")
train_img_dir = os.path.join(train_dir, "train_images")
test_img_dir = os.path.join(base_path, "test", "test_images")

# Automatically find the .json file inside train/
json_files = [f for f in os.listdir(train_dir) if f.endswith(".json")]
if not json_files:
    raise FileNotFoundError(f"No .json annotation file found in {train_dir}")

ann_json_path = os.path.join(train_dir, json_files[0])
print(f"Found annotation file: {ann_json_path}")
TARGET_SIZE = (256,256)
with open(ann_json_path, "r") as f:
    coco_data = json.load(f)

images_info = {img["id"]: img for img in coco_data["images"]}
anns_by_img = {img_id: [] for img_id in images_info}
for ann in coco_data["annotations"]:
    anns_by_img[ann["image_id"]].append(ann)
X_full = []
y_full = []
sorted_img_infos = sorted(coco_data["images"], key=lambda x: x["file_name"])


for img_info in sorted_img_infos:
    img_id = img_info["id"]
    file_name = img_info['file_name']
    orig_h, orig_w = img_info["height"], img_info["width"]
    img_path = os.path.join(train_img_dir, file_name)
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img_raw, channels=1, expand_animations=False
                        )
    img = tf.image.resize(img, TARGET_SIZE)
    img = (tf.cast(img, tf.float32) / 255.0).numpy()

    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for ann in anns_by_img.get(img_id, []):
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], color=1)
    mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    mask_resized = np.expand_dims(mask_resized, axis=-1).astype(np.float32)

    X_full.append(img)
    y_full.append(mask_resized)

X_full = np.array(X_full, dtype=np.float32)
y_full = np.array(X_full, dtype=np.float32)

x_test = []
for f_name in sorted(os.listdir(test_img_dir)):
    if f_name.lower().endswith((".png",".jpg",".jpeg")):
        img_raw = tf.io.read_file(os.path.join(test_img_dir, f_name))
        img = tf.image.decode_image(img_raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, TARGET_SIZE, method="bilinear")
        x_test.append((tf.cast(img, tf.float32) / 255.0).numpy())
x_test = np.array(x_test, dtype=np.float32)

print(f"Final X_train array shape: {X_full.shape}, Y_train shape{y_full.shape}")
print(f"Final X_test array shape: {X_full.shape}")

rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(X_full))
split_idx = int(len(X_full) * 0.8)

X_train, y_train = X_full[indices[:split_idx]], y_full[indices[:split_idx]]
X_val, y_val = X_full[indices[split_idx:]], y_full[indices[split_idx:]]

fig, axes = plt.subplots(2,5, figsize=(12,5))
for i in range(5):
    axes[0, i].imshow(X_train[i].squeeze(), cmap="gray")
    axes[0, i].set_title("H-Alpha Image")
    axes[0, i].axis("off")

    axes[1, i].imshow(y_train[i].squeeze(), cmap="gray")
    axes[1, i].set_title("Filament Mask")
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig("preview.png", dpi=150)

