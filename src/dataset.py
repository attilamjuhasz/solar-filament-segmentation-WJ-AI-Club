import json
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from pycocotools import mask as mask_utils


class SolarFilamentDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

        self.image_dir = (
            self.data_dir
            / "train"
            / "train_images"
        )

        annotation_file = (
            self.data_dir
            / "train"
            / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        )

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            self.data = json.load(f)

        self.images = self.data["images"]

        # Group annotations by image
        self.annotations = {}

        for annotation in self.data["annotations"]:

            image_id = annotation["image_id"]

            if image_id not in self.annotations:
                self.annotations[image_id] = []

            self.annotations[image_id].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_info = self.images[index]

        image_id = image_info["id"]

        filename = image_info["file_name"]

        image_path = self.image_dir / filename

        # Load grayscale image
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not find {image_path}"
            )

        height, width = image.shape

        # --------------------------------
        # Create masks
        # --------------------------------

        masks = []
        categories = []

        for annotation in self.annotations.get(
            image_id,
            []
        ):

            segmentation = annotation["segmentation"]

            # Polygon → RLE
            rles = mask_utils.frPyObjects(
                segmentation,
                height,
                width
            )

            rle = mask_utils.merge(rles)

            # RLE → binary mask
            mask = mask_utils.decode(rle)

            masks.append(mask)

            categories.append(
                annotation["category_id"]
            )

        # --------------------------------
        # Convert image to PyTorch tensor
        # --------------------------------

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image)

        # [H, W] → [1, H, W]
        image = image.unsqueeze(0)

        # --------------------------------
        # Convert masks
        # --------------------------------

        if len(masks) > 0:

            masks = np.stack(masks)

        else:

            masks = np.zeros(
                (0, height, width),
                dtype=np.uint8
            )

        masks = torch.from_numpy(masks)

        categories = torch.tensor(
            categories,
            dtype=torch.long
        )

        return {
            "image": image,
            "masks": masks,
            "categories": categories,
            "image_id": image_id,
            "filename": filename
        }