"""Create a tiny synthetic dataset with the same layout as the real one, so the
whole pipeline can be tested without downloading anything.

    python tests/make_fake_data.py fake_data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

SIZE = 2048


def fake_sun(rng: np.random.Generator):
    yy, xx = np.indices((SIZE, SIZE))
    cx, cy, r = SIZE / 2 + rng.uniform(-30, 30), SIZE / 2 + rng.uniform(-30, 30), 900
    rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    disk = rho <= 1
    img = np.where(disk, 170 * (0.6 + 0.4 * np.sqrt(np.clip(1 - rho**2, 0, 1))), 5.0)

    filaments = []
    for _ in range(rng.integers(3, 7)):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, 0.75) * r
        x, y = cx + dist * np.cos(ang), cy + dist * np.sin(ang)
        heading = rng.uniform(0, 2 * np.pi)
        pts = []
        for _ in range(rng.integers(8, 20)):
            heading += rng.normal(0, 0.3)
            x, y = x + 12 * np.cos(heading), y + 12 * np.sin(heading)
            pts.append((int(x), int(y)))
        cv2.polylines(mask, [np.array(pts)], False, 1, thickness=int(rng.integers(6, 16)))
        mask &= disk
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 300:
            continue
        img[mask > 0] *= 0.7
        poly = contour.reshape(-1, 2).astype(float)
        poly = np.vstack([poly, poly[:1]])  # closed path, like MAGFiLO
        filaments.append(poly.flatten().tolist())

    img = img + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), filaments


def main(out: str):
    rng = np.random.default_rng(0)
    root = Path(out)
    train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    for i in range(10):
        stem = f"201501{i + 10:02d}172714Mh"
        img, filaments = fake_sun(rng)
        cv2.imwrite(str(train_dir / f"{stem}.jpeg"), img)
        n_annotators = 2 if i % 3 == 0 else 1
        for a in range(n_annotators):
            image_id = f"0101{a + 1:02d}-{stem}"
            images.append({"id": image_id, "width": SIZE, "height": SIZE, "file_name": f"{stem}.jpg"})
            for j, poly in enumerate(filaments):
                annotations.append(
                    {"id": f"{image_id}-{j}", "image_id": image_id, "category_id": 1,
                     "segmentation": [poly], "area": 0.0, "spine": [], "bbox": [0, 0, 0, 0], "iscrowd": 0}
                )
    for i in range(3):
        img, _ = fake_sun(rng)
        cv2.imwrite(str(test_dir / f"201702{i + 10:02d}024112Bh.jpeg"), img)

    coco = {"info": {}, "licenses": [], "categories": [{"id": 1, "name": "Left", "supercategory": "filament"}],
            "images": images, "annotations": annotations}
    (root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(coco))
    print(f"wrote {len(images)} annotation sets / {len(annotations)} filaments to {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_data")
