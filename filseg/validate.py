"""Score a model on the validation split, re-using cached low-res inputs."""
from __future__ import annotations

import numpy as np

from .data import AnnotatedImage, group_by_stem
from .dataset import FilamentDataset
from .infer import predict_lowres, upsample
from .metrics import ImageScore, score_image
from .postprocess import PostParams, to_instances
from .rle import labels_to_rles


def predict_val_probs(model, dataset: FilamentDataset, device, tta: bool = False) -> dict[str, np.ndarray]:
    """One low-res probability map per image file (not per annotator)."""
    probs = {}
    for idx, item in enumerate(dataset.items):
        if item.stem not in probs:
            x, _ = dataset.load(idx)
            probs[item.stem] = predict_lowres(model, x, device, tta).astype(np.float16)
    return probs


def score_probs(
    probs: dict[str, np.ndarray], items: list[AnnotatedImage], params: PostParams
) -> list[ImageScore]:
    """Post-process each probability map once and score it against every
    annotator's version of that image."""
    scores = []
    for stem, group in group_by_stem(items).items():
        shape = (group[0].height, group[0].width)
        labels = to_instances(upsample(probs[stem].astype(np.float32), shape), params)
        pred = labels_to_rles(labels)
        scores.extend(score_image(item.image_id, item.rles(), pred) for item in group)
    return scores
