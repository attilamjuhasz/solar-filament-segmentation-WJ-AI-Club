#!/usr/bin/env python3
"""Submission Generator for Solar Filament Segmentation Challenge 2026.

Generates a strictly compliant, unquoted ASCII COCO compressed RLE submission.csv
at native 2048x2048 resolution for all 180 test observations.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.manifest import canonical_observation_id
from src.inference.rle import (
    audit_submission_and_manifest,
    compute_file_sha256,
    encode_instance,
    validate_submission_csv,
    write_submission_csv,
)


def extract_filaments_from_image(
    image_path: str,
    obs_id: str,
    disk_kernel: np.ndarray,
    bth_kernel: np.ndarray,
    seed_thresh: int = 50,
    low_thresh: int = 24,
    min_area: int = 500,
    min_eccentricity: float = 0.90,
    min_major_axis: float = 90.0,
    max_candidates_per_image: int = 10,
) -> List[Tuple[str, str]]:
    """Extract physical solar filament segmentations and return (filament_id, rle) pairs."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    if img.shape != NATIVE_IMAGE_SHAPE:
        raise ValueError(f"Image {image_path} shape {img.shape} != native {NATIVE_IMAGE_SHAPE}")

    # 1. Segment solar disk and erode limb boundary to remove outer flare/limb artifacts
    _, disk_mask = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    disk_eroded = cv2.erode(disk_mask, disk_kernel)

    # 2. Black top-hat transform to isolate absorption structures
    bth = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, bth_kernel)
    bth = cv2.bitwise_and(bth, bth, mask=disk_eroded)

    # 3. 8-connected Hysteresis thresholding
    high_mask = (bth >= seed_thresh)
    low_mask = (bth >= low_thresh)

    labeled_low, n_low = ndimage.label(low_mask, structure=np.ones((3, 3), dtype=bool))
    if n_low == 0:
        return []

    valid_labels = np.unique(labeled_low[high_mask])
    valid_labels = valid_labels[valid_labels != 0]
    if len(valid_labels) == 0:
        return []

    support = np.isin(labeled_low, valid_labels)
    lbl_supp, n_supp = ndimage.label(support, structure=np.ones((3, 3), dtype=bool))
    props = regionprops(lbl_supp, intensity_image=bth)

    scored_candidates = []
    for p in props:
        # Physical filament priors: minimum area, high elongation / spine length
        if p.area >= min_area and (p.eccentricity >= min_eccentricity or p.axis_major_length >= min_major_axis):
            sl = p.slice
            submask = np.zeros(img.shape, dtype=np.uint8)
            submask[sl] = (lbl_supp[sl] == p.label).astype(np.uint8)

            # Contrast score: mean absorption intensity weighted by log-scale area
            score = float(p.intensity_mean * np.log1p(p.area))
            scored_candidates.append((score, submask))

    if not scored_candidates:
        return []

    # Rank by contrast score and retain top K candidates
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    selected = scored_candidates[:max_candidates_per_image]

    rows = []
    for idx, (_, mask) in enumerate(selected, start=1):
        fid = f"{obs_id}_{idx}"
        rle = encode_instance(mask, expected_shape=NATIVE_IMAGE_SHAPE)
        rows.append((fid, rle))

    return rows


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_submission(
    test_dir: str,
    output_csv: str,
    max_candidates: int = 10,
) -> None:
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")))
    if not test_files:
        raise ValueError(f"No test images found in {test_dir}")

    print(f"Found {len(test_files)} test images in {test_dir}")
    print(f"Target submission path: {output_csv}")

    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))

    all_rows: List[Tuple[str, str]] = []
    detected_obs_count = 0
    t0 = time.time()

    for idx, path in enumerate(test_files, start=1):
        obs_id = canonical_observation_id(path)
        img_rows = extract_filaments_from_image(
            image_path=path,
            obs_id=obs_id,
            disk_kernel=disk_kernel,
            bth_kernel=bth_kernel,
            max_candidates_per_image=max_candidates
        )
        if img_rows:
            detected_obs_count += 1
            all_rows.extend(img_rows)

        if idx % 20 == 0 or idx == len(test_files):
            elapsed = time.time() - t0
            print(f"[{idx:3d}/{len(test_files)}] Processed. Current instances: {len(all_rows)} ({elapsed:.1f}s)")

    print(f"\nWriting {len(all_rows)} instances across {detected_obs_count}/{len(test_files)} observations...")
    write_submission_csv(output_csv, all_rows)
    print(f"Successfully saved {output_csv}")

    # Generate and save observation-completion manifest (Codex P2 finding #8)
    manifest_path = Path(output_csv).with_suffix(".manifest.json")
    obs_counts = {}
    for fid, _ in all_rows:
        obs = fid.rsplit("_", 1)[0]
        obs_counts[obs] = obs_counts.get(obs, 0) + 1

    expected_ids_sorted = sorted([canonical_observation_id(p) for p in test_files])
    csv_sha256 = compute_file_sha256(output_csv)
    manifest = {
        "submission_csv": str(output_csv),
        "csv_sha256": csv_sha256,
        "total_test_observations": len(expected_ids_sorted),
        "detected_observations_count": detected_obs_count,
        "abstained_observations_count": len(expected_ids_sorted) - detected_obs_count,
        "total_instances": len(all_rows),
        "observations": [
            {
                "observation_id": obs,
                "status": "processed" if obs in obs_counts else "abstained",
                "instance_count": obs_counts.get(obs, 0)
            }
            for obs in expected_ids_sorted
        ]
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved observation-completion manifest to {manifest_path}")

    # Strict audit
    print("\nRunning strict submission verification audit...")
    expected_ids = set(expected_ids_sorted)
    audit_report = audit_submission_and_manifest(
        csv_path=output_csv,
        manifest_path=manifest_path,
        expected_observation_ids=expected_ids,
        verify_rle_decoding=True
    )
    report = audit_report["csv_report"]
    print("\n=== Submission Validation Audit Report ===")
    print(f"Is Valid:                  {audit_report['is_valid']}")
    print(f"CSV SHA256:                {audit_report['csv_sha256']}")
    print(f"Total Rows (Instances):    {report['total_instances']}")
    print(f"Observations with Filaments: {report['unique_observations']} / {len(expected_ids)}")
    print(f"Observations Abstained:     {len(expected_ids) - report['unique_observations']}")
    print(f"Total Decoded Pixel Area:  {report['total_pixel_area']:,} px")
    print("==========================================")


def main():
    parser = argparse.ArgumentParser(description="Generate verified submission.csv for Solar Filament Segmentation 2026")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        help="Path to test images directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path to output submission.csv"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum candidates per observation"
    )
    args = parser.parse_args()

    generate_submission(
        test_dir=args.test_dir,
        output_csv=args.output,
        max_candidates=args.max_candidates
    )


if __name__ == "__main__":
    main()
