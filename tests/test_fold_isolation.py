import numpy as np
import pytest

from src.data.folds import (
    ObservationRecord,
    assign_stratified_group_folds,
    verify_fold_isolation,
)


def test_stratified_group_kfold_isolation():
    # Create 50 observations, each with 1-3 annotator variants
    records = []
    for i in range(50):
        day = (i % 28) + 1
        hour = i % 24
        obs_id = f"201501{day:02d}{hour:02d}0000Mh"
        variants = [f"010401-{obs_id}", f"010402-{obs_id}"]
        # Variable counts and areas
        count = float((i % 7) + 1)
        area = float((i * 1000) + 500)
        records.append(ObservationRecord(
            observation_id=obs_id,
            annotator_variants=variants,
            median_instance_count=count,
            median_mask_area=area
        ))

    fold_assignments = assign_stratified_group_folds(records, n_splits=5, random_state=2026)

    # Every canonical observation and variant must be assigned
    for r in records:
        assert r.observation_id in fold_assignments
        for var in r.annotator_variants:
            assert var in fold_assignments
            assert fold_assignments[var] == fold_assignments[r.observation_id]

    assert set(fold_assignments.values()) == {0, 1, 2, 3, 4}

    # Verify zero leakage
    assert verify_fold_isolation(fold_assignments, records) is True

    # Fold sizes for canonical observations should be approximately balanced
    canon_folds = [fold_assignments[r.observation_id] for r in records]
    counts = [canon_folds.count(i) for i in range(5)]
    for c in counts:
        assert 8 <= c <= 12


def test_codex_variant_leakage_regression():
    """Codex P1 finding #3: Records named 010401-20150125172714Mh and 010402-20150125172714Mh

    Must be assigned to the same fold, and verifier must catch any cross-fold variant split.
    """
    r1 = ObservationRecord("010401-20150125172714Mh", ["010401-20150125172714Mh"], 2.0, 1000.0)
    r2 = ObservationRecord("010402-20150125172714Mh", ["010402-20150125172714Mh"], 2.0, 1000.0)

    # 1. Assignment must assign identical fold
    assignments = assign_stratified_group_folds([r1, r2], n_splits=2)
    assert assignments[r1.observation_id] == assignments[r2.observation_id]
    assert verify_fold_isolation(assignments, [r1, r2], n_splits=2) is True

    # 2. Verifier must reject split across folds
    leaked_assignments = {
        "010401-20150125172714Mh": 1,
        "010402-20150125172714Mh": 4,
        "20150125172714Mh": 1,
    }
    with pytest.raises(ValueError, match="Multi-annotator fold leakage detected"):
        verify_fold_isolation(leaked_assignments, [r1, r2], n_splits=5)

    # 3. Verifier must reject empty assignments for non-empty records
    with pytest.raises(ValueError, match="fold_assignments is empty"):
        verify_fold_isolation({}, [r1, r2], n_splits=5)


def test_cross_observation_variant_reuse_rejection():
    """Verify Codex P2 finding #7: A variant mapping to a different canonical observation must be rejected."""
    # r1 is from 20150125, but claims variant from 20150126
    r1 = ObservationRecord(
        "20150125172714Mh",
        ["010401-20150126172714Mh"],  # Belongs to 20150126!
        median_instance_count=1.0,
        median_mask_area=500.0
    )

    # 1. Rejection during consolidation
    from src.data.folds import consolidate_canonical_records
    with pytest.raises(ValueError, match="Cross-observation variant mismatch"):
        consolidate_canonical_records([r1])

    # 2. Rejection during verify_fold_isolation
    valid_map = {
        "20150125172714Mh": 0,
        "010401-20150126172714Mh": 0
    }
    with pytest.raises(ValueError, match="Variant ownership violation"):
        verify_fold_isolation(valid_map, [r1], n_splits=5)



def test_real_dataset_fold_manifest_and_isolation():
    """Verify 5-fold stratification and isolation across all 707 canonical observations from train JSON."""
    import os
    import json
    from src.data.annotations import load_coco_annotations

    train_json = r"data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not os.path.exists(train_json):
        pytest.skip("Dataset not present locally")

    index = load_coco_annotations(train_json)
    records = []
    for canon_id, variants in index.by_observation.items():
        counts = [len(v.instances) for v in variants]
        areas = [sum(inst.area for inst in v.instances) for v in variants]
        records.append(ObservationRecord(
            observation_id=canon_id,
            annotator_variants=[v.annotator_image_id for v in variants],
            median_instance_count=float(np.median(counts)),
            median_mask_area=float(np.median(areas))
        ))

    assert len(records) == 707
    fold_assignments = assign_stratified_group_folds(records, n_splits=5, random_state=2026)
    assert verify_fold_isolation(fold_assignments, records, n_splits=5) is True

    # Check fold distribution: each fold should have ~141 observations
    counts = [sum(1 for r in records if fold_assignments[r.observation_id] == i) for i in range(5)]
    for c in counts:
        assert 135 <= c <= 150
