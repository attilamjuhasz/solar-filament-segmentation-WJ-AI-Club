import numpy as np
import pytest

from src.contracts import NATIVE_IMAGE_SHAPE
from src.evaluation.competition_adapter import evaluate_entry_pq, evaluate_dataset_pq


def test_perfect_single_match():
    # 1 GT, 1 Pred perfectly aligned
    m = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m[100:200, 100:200] = 1

    total_iou, tp, fp, fn, o2m, m2o = evaluate_entry_pq([m], [m])
    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert abs(total_iou - 1.0) < 1e-6

    stats = evaluate_dataset_pq([([m], [m])])
    assert abs(stats.pq - 1.0) < 1e-6
    assert abs(stats.sq - 1.0) < 1e-6
    assert abs(stats.rq - 1.0) < 1e-6


def test_iou_threshold_boundary():
    # Construct GT and Pred with exact overlap
    # Overlap = 100 pixels, Total union = 200 pixels -> IoU = 0.50 exactly
    # Under strict > 0.5, IoU = 0.50 is NOT a hit!
    gt = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    pred = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)

    gt[0, :150] = 1    # 150 px
    pred[0, 50:200] = 1  # 150 px
    # Intersection: 50..149 -> 100 px
    # Union: 0..199 -> 200 px. IoU = 100/200 = 0.50
    total_iou, tp, fp, fn, _, _ = evaluate_entry_pq([gt], [pred])
    assert tp == 0
    assert fp == 1
    assert fn == 1
    assert total_iou == 0.0

    # Now make intersection 101 / union 199 -> IoU = 101/199 = 0.5075 > 0.5
    pred2 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    pred2[0, 49:199] = 1  # intersection: 49..149 -> 101 px, union: 0..198 -> 199 px
    total_iou2, tp2, fp2, fn2, _, _ = evaluate_entry_pq([gt], [pred2])
    assert tp2 == 1
    assert fp2 == 0
    assert fn2 == 0
    assert abs(total_iou2 - (101.0 / 199.0)) < 1e-6


def test_many_to_one_merging_penalty():
    # Two disjoint GT filaments, one merged prediction covering both
    gt1 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    gt1[100:150, 100:150] = 1  # 2500 px

    gt2 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    gt2[200:250, 200:250] = 1  # 2500 px

    merged_pred = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    merged_pred[100:150, 100:150] = 1
    merged_pred[200:250, 200:250] = 1  # 5000 px

    # IoU(gt1, pred) = 2500 / 5000 = 0.50 (fails > 0.5)
    # IoU(gt2, pred) = 2500 / 5000 = 0.50 (fails > 0.5)
    total_iou, tp, fp, fn, o2m, m2o = evaluate_entry_pq([gt1, gt2], [merged_pred])
    assert tp == 0
    assert fp == 1
    assert fn == 2
    assert m2o == 1  # detected as many-to-one merger


def test_empty_cases():
    total_iou, tp, fp, fn, _, _ = evaluate_entry_pq([], [])
    assert tp == 0 and fp == 0 and fn == 0

    stats = evaluate_dataset_pq([([], [])])
    assert stats.pq == 0.0
