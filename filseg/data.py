"""Loading MAGFiLO H-alpha images and their COCO-style filament annotations."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

IMG_SIZE = 2048
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class AnnotatedImage:
    """One annotation set: a single annotator's filaments for a single image.

    The same image file can appear several times with different annotators,
    e.g. ids ``010401-20160920230134Lh`` and ``010402-20160920230134Lh``.
    """

    image_id: str
    file_name: str
    height: int = IMG_SIZE
    width: int = IMG_SIZE
    # One entry per filament; each entry is a list of polygons [x0, y0, x1, y1, ...].
    filaments: list[list[list[float]]] = field(default_factory=list)

    @property
    def stem(self) -> str:
        """Image name without extension, e.g. ``20160920230134Lh``. Used in submissions."""
        return Path(self.file_name).stem

    def rles(self) -> list[dict]:
        """One COCO RLE per filament."""
        return [polygons_to_rle(polys, self.height, self.width) for polys in self.filaments]

    def union_mask(self) -> np.ndarray:
        """Binary HxW uint8 mask of all filaments."""
        if not self.filaments:
            return np.zeros((self.height, self.width), np.uint8)
        return mask_utils.decode(mask_utils.merge(self.rles(), intersect=False))


def polygons_to_rle(polygons: list[list[float]], height: int, width: int) -> dict:
    polygons = [p for p in polygons if len(p) >= 6]
    return mask_utils.merge(mask_utils.frPyObjects(polygons, height, width), intersect=False)


def load_annotations(json_path: str | Path) -> list[AnnotatedImage]:
    with open(json_path) as f:
        coco = json.load(f)
    by_id: dict[str, AnnotatedImage] = {}
    for img in coco["images"]:
        by_id[str(img["id"])] = AnnotatedImage(
            image_id=str(img["id"]),
            file_name=img["file_name"],
            height=int(img.get("height", IMG_SIZE)),
            width=int(img.get("width", IMG_SIZE)),
        )
    for ann in coco["annotations"]:
        img = by_id.get(str(ann["image_id"]))
        polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
        if img is not None and polygons:
            img.filaments.append(polygons)
    return list(by_id.values())


def index_images(images_dir: str | Path) -> dict[str, Path]:
    """Map image stem -> path, so ``.jpg`` vs ``.jpeg`` mismatches don't matter."""
    return {p.stem: p for p in sorted(Path(images_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTS}


def read_image(path: str | Path) -> np.ndarray:
    """Read an H-alpha observation as a single-channel uint8 image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def find_data_root(search_dir: str | Path = ".") -> Path:
    """Find the dataset folder (the one holding ``train/`` and ``test/``) under ``search_dir``."""
    search_dir = Path(search_dir)
    for candidate in [search_dir, *search_dir.glob("**/")]:
        if (candidate / "train" / "train_images").is_dir():
            return candidate
    raise FileNotFoundError(f"No train/train_images folder found under {search_dir}")


@dataclass
class DataPaths:
    root: Path

    @property
    def train_images(self) -> Path:
        return self.root / "train" / "train_images"

    @property
    def test_images(self) -> Path:
        return self.root / "test" / "test_images"

    @property
    def train_json(self) -> Path:
        return next((self.root / "train").glob("*.json"))


def group_by_stem(items: list[AnnotatedImage]) -> dict[str, list[AnnotatedImage]]:
    """Image stem -> all annotation sets (annotators) of that image."""
    groups: dict[str, list[AnnotatedImage]] = defaultdict(list)
    for item in items:
        groups[item.stem].append(item)
    return groups


def split_by_image(
    items: list[AnnotatedImage], val_frac: float = 0.15, seed: int = 0
) -> tuple[list[AnnotatedImage], list[AnnotatedImage]]:
    """Train/val split grouped by image file, so no image appears on both sides
    through a different annotator."""
    stems = sorted(group_by_stem(items))
    random.Random(seed).shuffle(stems)
    n_val = max(1, round(len(stems) * val_frac))
    val_stems = set(stems[:n_val])
    train = [i for i in items if i.stem not in val_stems]
    val = [i for i in items if i.stem in val_stems]
    return train, val
