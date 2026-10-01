from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.manifest import canonical_observation_id


@dataclass
class ObservationRecord:
    """Record summarizing an independent physical observation for fold assignment."""
    observation_id: str
    annotator_variants: List[str]
    median_instance_count: float
    median_mask_area: float
    disk_area: float = 2048.0 * 2048.0


def consolidate_canonical_records(
    records: Sequence[ObservationRecord]
) -> List[ObservationRecord]:
    """Consolidate records by canonical observation ID to prevent multi-annotator leakage.
    
    If the input contains separate records for annotator variants of the same observation
    (e.g. '010401-20150125172714Mh' and '010402-20150125172714Mh'), this aggregates them
    into a single canonical ObservationRecord with all annotator variants merged.
    """
    by_canon: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "variants": set(),
        "counts": [],
        "areas": []
    })

    for r in records:
        canon_id = canonical_observation_id(r.observation_id)
        entry = by_canon[canon_id]
        entry["variants"].add(r.observation_id)
        for var in r.annotator_variants:
            var_canon = canonical_observation_id(var)
            if var_canon != canon_id:
                raise ValueError(
                    f"Cross-observation variant mismatch: record '{r.observation_id}' (canonical '{canon_id}') "
                    f"claims variant '{var}' (which maps to canonical '{var_canon}')."
                )
            entry["variants"].add(var)
        entry["counts"].append(r.median_instance_count)
        entry["areas"].append(r.median_mask_area)

    consolidated = []
    for canon_id, entry in sorted(by_canon.items()):
        consolidated.append(ObservationRecord(
            observation_id=canon_id,
            annotator_variants=sorted(list(entry["variants"])),
            median_instance_count=float(np.median(entry["counts"])),
            median_mask_area=float(np.median(entry["areas"])),
        ))

    return consolidated


def compute_stratification_labels(
    records: Sequence[ObservationRecord],
    n_splits: int = 5,
    n_bins: int = 3
) -> np.ndarray:
    """Compute combined (count_bin : area_bin) stratification strata with rare-strata merging."""
    n_samples = len(records)
    if n_samples < n_splits * 2:
        return np.zeros(n_samples, dtype=np.int32)

    counts = np.array([r.median_instance_count for r in records], dtype=np.float64)
    areas = np.array([r.median_mask_area for r in records], dtype=np.float64)
    log_areas = np.log1p(areas)

    # Bin counts and areas into quantiles
    try:
        count_bins = pd.qcut(counts, q=n_bins, labels=False, duplicates="drop")
        area_bins = pd.qcut(log_areas, q=n_bins, labels=False, duplicates="drop")
        composite = count_bins.astype(str) + "_" + area_bins.astype(str)
    except Exception:
        return np.zeros(n_samples, dtype=np.int32)

    # Count occurrences of each stratum
    label_series = pd.Series(composite)
    counts_by_label = label_series.value_counts()

    # Merge rare strata (count < n_splits) into a common fallback stratum
    merged_labels = label_series.apply(
        lambda lbl: lbl if counts_by_label[lbl] >= n_splits else "rare_merged"
    )

    # If the rare_merged stratum itself has fewer than n_splits items, merge into most common stratum
    merged_counts = merged_labels.value_counts()
    if "rare_merged" in merged_counts and merged_counts["rare_merged"] < n_splits:
        # Find the most frequent non-rare stratum
        non_rare = [k for k in merged_counts.index if k != "rare_merged"]
        if non_rare:
            most_common = non_rare[0]
            merged_labels = merged_labels.replace({"rare_merged": most_common})
        else:
            # All strata are rare; assign single uniform stratum 0
            return np.zeros(n_samples, dtype=np.int32)

    unique_labels, inverse = np.unique(merged_labels, return_inverse=True)
    return inverse


def assign_stratified_group_folds(
    records: Sequence[ObservationRecord],
    n_splits: int = 5,
    random_state: int = 2026
) -> Dict[str, int]:
    """Assign observations to n_splits leakage-safe folds using StratifiedGroupKFold.
    
    Guarantees:
    - Enforces canonical observation grouping: all annotator variants of the same
      observation (e.g. '010401-20150125172714Mh' and '010402-20150125172714Mh')
      are guaranteed to land in the EXACT same fold.
    - Resolves both canonical IDs and all variant IDs into the fold dictionary.
    - Complete assignment: every record is assigned a fold ID in range(0, n_splits).
    - Stratified by joint filament count and area distribution with rare-strata merging.
    """
    if len(records) == 0:
        return {}

    # 1. Consolidate into unique canonical observations
    canon_records = consolidate_canonical_records(records)
    n_obs = len(canon_records)

    strata = compute_stratification_labels(canon_records, n_splits=n_splits)
    groups = np.arange(n_obs)  # Each canonical record is an independent group

    effective_splits = min(n_splits, n_obs)
    if effective_splits < 2:
        # All assigned to fold 0 if insufficient observations, populating canonical and variant keys
        fallback_assignments: Dict[str, int] = {}
        for r in records:
            canon_id = canonical_observation_id(r.observation_id)
            fallback_assignments[canon_id] = 0
            fallback_assignments[r.observation_id] = 0
            for var in r.annotator_variants:
                fallback_assignments[var] = 0
        return fallback_assignments

    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    dummy_X = np.zeros((n_obs, 1))

    fold_assignments: Dict[str, int] = {}
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(dummy_X, strata, groups)):
        for idx in val_idx:
            crec = canon_records[idx]
            # Assign canonical ID
            fold_assignments[crec.observation_id] = fold
            # Assign all associated annotator variants to the exact same fold
            for var in crec.annotator_variants:
                fold_assignments[var] = fold

    # Double check complete assignment for all raw records
    for r in records:
        canon_id = canonical_observation_id(r.observation_id)
        if canon_id not in fold_assignments:
            raise RuntimeError(f"Observation {canon_id} was missed during fold assignment")
        assigned_fold = fold_assignments[canon_id]
        fold_assignments[r.observation_id] = assigned_fold
        for var in r.annotator_variants:
            fold_assignments[var] = assigned_fold

    return fold_assignments


def verify_fold_isolation(
    fold_assignments: Dict[str, int],
    records: Sequence[ObservationRecord],
    n_splits: int = 5
) -> bool:
    """Verify complete absence of observation or annotator variant leakage across folds.
    
    Audits:
    1. If records is non-empty, fold_assignments cannot be empty.
    2. Every record and all its annotator variants must be present in fold_assignments.
    3. All variants belonging to the same canonical observation MUST have identical fold assignment.
    4. Canonical observations must be strictly partitioned with zero intersection between folds.
    5. All assigned fold IDs must be within range(0, n_splits).
    6. Variants declared by a record must legitimately map to its canonical observation ID.
    """
    if len(records) == 0:
        return True

    if not fold_assignments:
        raise ValueError("fold_assignments is empty for non-empty records sequence")

    # Map each canonical observation to the set of folds it was assigned to
    obs_to_folds: Dict[str, Set[int]] = defaultdict(set)
    assigned_folds: Set[int] = set()

    for r in records:
        canon_id = canonical_observation_id(r.observation_id)
        
        # Check that record ID is in assignments
        if r.observation_id not in fold_assignments:
            raise ValueError(f"Record {r.observation_id} missing from fold assignments")
        
        rec_fold = fold_assignments[r.observation_id]
        if not (0 <= rec_fold < n_splits):
            raise ValueError(f"Invalid fold ID {rec_fold} for record {r.observation_id}; expected [0, {n_splits})")

        obs_to_folds[canon_id].add(rec_fold)
        assigned_folds.add(rec_fold)

        # Check all annotator variants of this record
        for var in r.annotator_variants:
            var_canon = canonical_observation_id(var)
            if var_canon != canon_id:
                raise ValueError(
                    f"Variant ownership violation: record '{r.observation_id}' (canonical '{canon_id}') "
                    f"declares variant '{var}' belonging to canonical '{var_canon}'"
                )
            if var not in fold_assignments:
                raise ValueError(f"Variant {var} of {canon_id} missing from fold assignments")
            var_fold = fold_assignments[var]
            obs_to_folds[canon_id].add(var_fold)

    # Verify each canonical observation has exactly one assigned fold
    leakage_detected = []
    for canon_id, folds in obs_to_folds.items():
        if len(folds) > 1:
            leakage_detected.append(f"{canon_id} assigned to multiple folds: {sorted(list(folds))}")

    if leakage_detected:
        raise ValueError("Multi-annotator fold leakage detected:\n" + "\n".join(leakage_detected))

    # Cross-fold partition check
    for f in assigned_folds:
        val_obs = {obs for obs, folds in obs_to_folds.items() if f in folds}
        train_obs = {obs for obs, folds in obs_to_folds.items() if f not in folds}
        overlap = val_obs.intersection(train_obs)
        if overlap:
            raise ValueError(f"Fold {f} has observation partition overlap: {overlap}")

    return True


def save_folds_manifest(
    fold_assignments: Dict[str, int],
    records: Sequence[ObservationRecord],
    output_path: str
) -> None:
    """Persist fold assignments as CSV for guaranteed experiment reproducibility."""
    rows = []
    seen_variants = set()

    canon_records = consolidate_canonical_records(records)
    for crec in canon_records:
        canon_id = crec.observation_id
        f = fold_assignments[canon_id]
        for var in crec.annotator_variants:
            if var not in seen_variants:
                seen_variants.add(var)
                rows.append({
                    "observation_id": canon_id,
                    "annotator_image_id": var,
                    "fold": f,
                    "median_instance_count": crec.median_instance_count,
                    "median_mask_area": crec.median_mask_area
                })
    
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def compute_path_sha256(filepath: str | Path) -> str:
    """Compute lowercase SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def freeze_folds_manifest(
    records: Sequence[ObservationRecord],
    annotations_path: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
    seed: int = 2026,
) -> Tuple[Dict[str, int], str]:
    """Generate, verify, and freeze reproducible folds with dataset hash and manifest checksum.
    
    Returns:
        (fold_assignments, manifest_sha256)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_sha256 = compute_path_sha256(annotations_path)
    fold_assignments = assign_stratified_group_folds(records, n_splits=n_splits, random_state=seed)
    verify_fold_isolation(fold_assignments, records, n_splits=n_splits)

    csv_path = output_dir / "folds.csv"
    save_folds_manifest(fold_assignments, records, str(csv_path))
    csv_sha256 = compute_path_sha256(csv_path)

    canon_records = consolidate_canonical_records(records)
    fold_counts: Dict[str, int] = {}
    for crec in canon_records:
        f = str(fold_assignments[crec.observation_id])
        fold_counts[f] = fold_counts.get(f, 0) + 1

    manifest = {
        "annotations_path": str(annotations_path),
        "annotations_sha256": annotations_sha256,
        "n_splits": n_splits,
        "seed": seed,
        "total_canonical_observations": len(canon_records),
        "fold_counts": fold_counts,
        "folds_csv": str(csv_path.name),
        "folds_csv_sha256": csv_sha256,
        "assignments": fold_assignments,
    }

    manifest_path = output_dir / "folds_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    manifest_sha256 = compute_path_sha256(manifest_path)
    return fold_assignments, manifest_sha256


def load_frozen_folds_manifest(
    manifest_path: str | Path,
    verify_annotations_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, int], str]:
    """Load pre-computed frozen fold assignments and verify integrity."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Folds manifest not found at {manifest_path}")

    manifest_sha256 = compute_path_sha256(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verify_annotations_path is not None:
        curr_sha256 = compute_path_sha256(verify_annotations_path)
        expected_sha256 = data.get("annotations_sha256")
        if not expected_sha256:
            raise ValueError("Manifest lacks annotations_sha256 metadata")
        if curr_sha256 != expected_sha256:
            raise ValueError(
                f"Annotation SHA256 mismatch! Expected {expected_sha256}, got {curr_sha256}"
            )

    # Validate folds.csv if present
    csv_filename = data.get("folds_csv")
    csv_sha256 = data.get("folds_csv_sha256")
    if csv_filename and csv_sha256:
        csv_file = manifest_path.parent / csv_filename
        if csv_file.is_file():
            actual_csv_sha = compute_path_sha256(csv_file)
            if actual_csv_sha != csv_sha256:
                raise ValueError(
                    f"Folds CSV SHA256 mismatch! Expected {csv_sha256}, got {actual_csv_sha}"
                )

    assignments: Dict[str, int] = data["assignments"]
    n_splits = data.get("n_splits", 5)

    # Validate assignment types, bounds, and pairwise alias consistency
    for k, v in assignments.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v >= n_splits:
            raise ValueError(f"Invalid fold assignment {v} for key '{k}'; must be integer in [0, {n_splits - 1}]")
        canon_k = canonical_observation_id(k)
        if canon_k in assignments and assignments[canon_k] != v:
            raise ValueError(
                f"Pairwise alias inconsistency: variant '{k}' assigned to fold {v}, "
                f"but canonical observation '{canon_k}' assigned to fold {assignments[canon_k]}"
            )

    return assignments, manifest_sha256


# Recorded validation observations explored during preliminary diagnostic runs
EXPLORED_VAL_OBSERVATIONS: Tuple[str, ...] = (
    "20110109104734Ch",
    "20110306082634Lh",
    "20110312082634Lh",
    "20110317082654Uh",
    "20110322005814Mh",
)


def get_deterministic_fold_partitions(
    fold_assignments: Dict[str, int],
    target_fold: int = 0,
    seed: int = 2026,
    num_tuning: int = 10,
    num_confirmation: int = 10,
    use_migrated: bool = True,
    migrated_artifact_path: Union[str, Path] = "artifacts/partitions_migrated_v1.json",
) -> Dict[str, List[str]]:
    """Deterministically partition validation observations of target_fold into explored, tuning, and confirmation sets.

    Guarantees:
    - Partitions operate strictly on unique canonical physical observations (no alias duplicate weighting).
    - If migrated partition artifact exists and target_fold=0, loads frozen migrated v1 partitions (10 tuning, 9 confirmation).
    - Explored IDs are segregated.
    - Tuning IDs and Confirmation IDs are completely pairwise disjoint.
    - Partition is independent of any evaluation limit argument.
    """
    migrated_path = Path(migrated_artifact_path)
    if use_migrated and target_fold == 0 and migrated_path.is_file():
        import json
        with open(migrated_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        tuning = list(pdata["tuning"]["canonical_observation_ids"])
        confirmation = list(pdata["confirmation"]["canonical_observation_ids"])
        explored = list(pdata["explored"]["canonical_observation_ids"])
        all_val = sorted(list(set(canonical_observation_id(k) for k, f in fold_assignments.items() if f == target_fold)))
        return {
            "explored": explored,
            "tuning": tuning,
            "confirmation": confirmation,
            "all_validation": all_val,
        }

    # Canonical observation universe
    canonical_val_obs = sorted(list({canonical_observation_id(k) for k, f in fold_assignments.items() if f == target_fold}))
    explored_set = {canonical_observation_id(x) for x in EXPLORED_VAL_OBSERVATIONS}
    explored = [obs for obs in canonical_val_obs if obs in explored_set]
    fresh_obs = [obs for obs in canonical_val_obs if obs not in explored_set]

    rng = np.random.RandomState(seed)
    shuffled_fresh = fresh_obs.copy()
    rng.shuffle(shuffled_fresh)

    n_tune = min(num_tuning, len(shuffled_fresh))
    tuning = sorted(shuffled_fresh[:n_tune])
    remaining_fresh = shuffled_fresh[n_tune:]
    n_conf = min(num_confirmation, len(remaining_fresh))
    confirmation = sorted(remaining_fresh[:n_conf])

    return {
        "explored": explored,
        "tuning": tuning,
        "confirmation": confirmation,
        "all_validation": canonical_val_obs,
    }



