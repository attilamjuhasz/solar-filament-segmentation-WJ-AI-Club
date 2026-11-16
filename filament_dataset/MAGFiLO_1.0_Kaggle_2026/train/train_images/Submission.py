import os
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
import tensorflow as tf

# 1. Load your trained U-Net model
model = tf.keras.models.load_model("filament_unet.keras", compile=False)

test_img_dir = "filament_dataset/MAGFiLO_1.0_Kaggle_2026/test/test_images"
test_files = sorted([
    f
    for f in os.listdir(test_img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

rows = []
MIN_FILAMENT_PIXELS = 50  # Filters out tiny single-pixel noise blobs at 2048x2048

print(f"Generating predictions for {len(test_files)} test images...")

for idx, file_name in enumerate(test_files):
  img_path = os.path.join(test_img_dir, file_name)
  image_id = os.path.splitext(file_name)[0]

  # 2. Preprocess image to match training (256x256 grayscale normalized)
  orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  input_img = cv2.resize(orig_img, (256, 256)).astype(np.float32) / 255.0
  input_batch = np.expand_dims(input_img, axis=(0, -1))

  # 3. Predict mask and resize back to the required 2048x2048 competition size
  pred_prob = model.predict(input_batch, verbose=0)[0, :, :, 0]
  binary_mask_256 = (pred_prob > 0.5).astype(np.uint8)
  mask_2048 = cv2.resize(
      binary_mask_256, (2048, 2048), interpolation=cv2.INTER_NEAREST
  )

  # 4. Split the image mask into individual connected filaments
  num_labels, labels_im = cv2.connectedComponents(mask_2048)

  filament_count = 0
  # Label 0 is background, so loop from 1 to num_labels - 1
  for label_id in range(1, num_labels):
    single_filament = (labels_im == label_id).astype(np.uint8)

    # Skip tiny false-positive speckles
    if np.sum(single_filament) < MIN_FILAMENT_PIXELS:
      continue

    filament_count += 1
    filament_id = f"{image_id}_{filament_count}"

    # 5. Encode using pycocotools (requires Fortran-contiguous uint8 array)
    fortran_mask = np.asfortranarray(single_filament)
    rle = mask_utils.encode(fortran_mask)
    rle_counts = rle["counts"].decode("utf-8")

    rows.append({"filament_id": filament_id, "segmentation_rle": rle_counts})

  # Fallback: if no filaments were detected in an image, check sample_submission.csv
  # to verify whether empty images should be omitted or given an empty RLE.
  if (idx + 1) % 25 == 0 or (idx + 1) == len(test_files):
    print(f"Processed {idx + 1}/{len(test_files)} images...")

# 6. Save to submission.csv
submission_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
submission_df.to_csv("submission.csv", index=False)
print(f"Done! Saved {len(submission_df)} predicted filaments to submission.csv")
