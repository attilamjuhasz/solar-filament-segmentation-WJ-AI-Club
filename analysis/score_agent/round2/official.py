"""VERBATIM port of the scoring functions from the official self-evaluation notebook
(nicknnamsa/solar-filament-segmentation notebooks/self-evaluation-notebook.ipynb, cells 3-4).
Only change: get_pq_score_counts = get_pq_score with the counts returned alongside (logic untouched).
"""
import pandas as pd
import numpy as np
import torch
from pycocotools import mask as mask_util


def get_overlap_matrices(
        gt_layers: torch.Tensor,
        pred_layers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())

    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)

    union = gt_areas + pred_areas - intersection

    iou_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        intersection / union,
    )

    dice_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )

    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    gt_matches_per_pred = hit_matrix.sum(dim=0)
    return (gt_matches_per_pred == 0).sum().item()


def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    pred_matches_per_gt = hit_matrix.sum(dim=1)
    return (pred_matches_per_gt == 0).sum().item()


def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    # decode -> (H, W, n_masks), transpose to (n_masks, H, W)
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(
        annotator_image: str,
        gt_annotator_image_ids: pd.Series,
        pred_image_ids: pd.Series,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    image_id = annotator_image.split("-", maxsplit=1)[1]

    # (n_gt, H, W) stack with one layer per GT filament of this annotator-image
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))

    # (n_pred, H, W) stack with one layer per predicted filament of this image
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))

    n_gt = len(gt_rles)
    n_pred = len(pred_rles)

    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)

    return iou_matrix, dice_matrix, n_gt, n_pred


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]

    annotator_images = gt_annotator_image_ids.unique()

    overlap_df = pd.DataFrame({"annotator_image": annotator_images})

    results = overlap_df["annotator_image"].apply(
        process_entry,
        gt_annotator_image_ids=gt_annotator_image_ids,
        pred_image_ids=pred_image_ids,
        gt_df=gt_df,
        pred_df=pred_df,
    )

    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df[
            "n_pred"] = zip(*results)

    return overlap_df


def get_pq_score(overlap_df: pd.DataFrame) -> float:
    return get_pq_score_counts(overlap_df)[0]


def get_pq_score_counts(overlap_df: pd.DataFrame):
    iou_threshold: float = 0.5

    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            fp_count += n_pred
            continue

        if n_pred == 0:
            fn_count += n_gt
            continue

        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())

        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count

    if denominator > 0:
        pq_score = sum(tp_iou_scores) / denominator
    else:
        pq_score = 0.0

    return pq_score, dict(S=sum(tp_iou_scores), tp=tp_count, fp=fp_count, fn=fn_count)
