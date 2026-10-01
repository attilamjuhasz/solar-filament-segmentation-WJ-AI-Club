#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from src.data.manifest import canonical_observation_id
from src.inference.rle import audit_submission_and_manifest, validate_submission_csv


def main():
    parser = argparse.ArgumentParser(description="Strict Kaggle submission.csv validator for Solar Filament Challenge 2026")
    parser.add_argument("csv_path", type=str, help="Path to submission.csv")
    parser.add_argument("--skip-rle-decode", action="store_true", help="Skip full 2048x2048 RLE decode check")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test images (*.jpeg) to verify observation IDs")
    parser.add_argument("--manifest", type=str, default=None, help="Optional observation-completion manifest JSON to verify completion status")
    args = parser.parse_args()

    expected_obs = None
    if args.test_dir:
        test_files = glob.glob(os.path.join(args.test_dir, "*.jpeg")) + glob.glob(os.path.join(args.test_dir, "*.png"))
        expected_obs = {canonical_observation_id(f) for f in test_files}
        print(f"Loaded {len(expected_obs)} expected test observations from {args.test_dir}")

    try:
        if args.manifest:
            # Full joint audit of CSV and completion manifest
            audit_report = audit_submission_and_manifest(
                csv_path=args.csv_path,
                manifest_path=args.manifest,
                expected_observation_ids=expected_obs,
                verify_rle_decoding=not args.skip_rle_decode
            )
            report = dict(audit_report["csv_report"])
            if expected_obs is not None:
                report["expected_observations_count"] = len(expected_obs)
                report["detected_observations_count"] = report["unique_observations"]
                report["zero_detection_observations_count"] = len(expected_obs) - report["unique_observations"]
            manifest_summary = dict(audit_report)
            manifest_summary.pop("csv_report", None)
            report["manifest_audit"] = manifest_summary
            print(f"Verified observation-completion manifest: {args.manifest}")
        else:
            report = validate_submission_csv(
                args.csv_path,
                expected_observation_ids=expected_obs,
                verify_rle_decoding=not args.skip_rle_decode
            )
            if expected_obs is not None:
                report["expected_observations_count"] = len(expected_obs)
                report["detected_observations_count"] = report["unique_observations"]
                report["zero_detection_observations_count"] = len(expected_obs) - report["unique_observations"]

        print("SUCCESS: submission.csv is 100% valid!")
        print(json.dumps(report, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
