import json
import os
import zipfile
import cv2
import numpy as np
import tensorflow as tf
from creatingModel import unet_model

model = unet_model(n_classes=1, img_height=256, img_width=256, img_channels=1)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)
# 1. Find and extract zip
current_dir = os.getcwd()
zip_name = next(
    (
        f
        for f in os.listdir(current_dir)
        if "filament" in f.lower() and f.endswith(".zip")
    ),
    None,
)
if not zip_name:
  raise FileNotFoundError("Could not find dataset .zip file!")

extract_dir = "filament_dataset"
if not os.path.exists(extract_dir):
  with zipfile.ZipFile(zip_name, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

base_path = os.path.join(extract_dir, "MAGFiLO_1.0_Kaggle_2026")
train_dir = os.path.join(base_path, "train")
train_img_dir = os.path.join(train_dir, "train_images")

json_files = [f for f in os.listdir(train_dir) if f.endswith(".json")]
ann_json_path = os.path.join(train_dir, json_files[0])
print(f"Found annotation file: {ann_json_path}")

TARGET_SIZE = (256, 256)
with open(ann_json_path, "r") as f:
  coco_data = json.load(f)

anns_by_img = {img["id"]: [] for img in coco_data["images"]}
for ann in coco_data["annotations"]:
  anns_by_img[ann["image_id"]].append(ann)

sorted_img_infos = sorted(coco_data["images"], key=lambda x: x["file_name"])


# 2. Memory-safe generator: loads only `batch_size` images at a time
class FilamentDataGenerator(tf.keras.utils.Sequence):

  def __init__(self, img_infos, batch_size=4, shuffle=True):
    self.img_infos = img_infos
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.indices = np.arange(len(self.img_infos))
    self.on_epoch_end()

  def __len__(self):
    return int(np.ceil(len(self.img_infos) / self.batch_size))

  def on_epoch_end(self):
    if self.shuffle:
      np.random.shuffle(self.indices)

  def __getitem__(self, idx):
    batch_indices = self.indices[
        idx * self.batch_size : (idx + 1) * self.batch_size
    ]
    batch_X, batch_y = [], []

    for i in batch_indices:
      info = self.img_infos[i]
      img_path = os.path.join(train_img_dir, info["file_name"])

      # Read grayscale image with OpenCV (much lighter than tf.io in loops)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, TARGET_SIZE).astype(np.float32) / 255.0
      img = np.expand_dims(img, axis=-1)

      # Build binary mask
      mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
      for ann in anns_by_img.get(info["id"], []):
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
          for poly in seg:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [pts], color=1)

      mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
      mask = np.expand_dims(mask, axis=-1).astype(np.float32)

      batch_X.append(img)
      batch_y.append(mask)

    return np.array(batch_X, dtype=np.float32), np.array(
        batch_y, dtype=np.float32
    )


# 3. Split metadata (zero RAM used!)
rng = np.random.default_rng(seed=42)
perm = rng.permutation(len(sorted_img_infos))
split_idx = int(len(sorted_img_infos) * 0.8)

train_infos = [sorted_img_infos[i] for i in perm[:split_idx]]
val_infos = [sorted_img_infos[i] for i in perm[split_idx:]]

train_gen = FilamentDataGenerator(train_infos, batch_size=4, shuffle=True)
val_gen = FilamentDataGenerator(val_infos, batch_size=4, shuffle=False)

print(
    f"Training on {len(train_infos)} images, validating on"
    f" {len(val_infos)} images."
)

# 4. Train and save
model.fit(train_gen, epochs=4, validation_data=val_gen)
model.save("filament_unet.keras")
