import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class SolarDataset(Dataset):
    def __init__(self, image_dir, ann_file, img_size=(512, 512)):
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Safely fetch image info handling both string and integer key types
        if img_id in self.coco.imgs:
            img_info = self.coco.imgs[img_id]
        elif int(img_id) in self.coco.imgs:
            img_info = self.coco.imgs[int(img_id)]
        else:
            img_info = self.coco.imgs[str(img_id)]
            
        image_path = os.path.join(self.image_dir, img_info['file_name'])

        # 2. Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]

        # 3. Generate binary mask from all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m)

        # 4. Resize image and mask to target training size
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 5. Normalize and convert to PyTorch tensors
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)) # (C, H, W)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)

        return image_tensor, mask_tensor