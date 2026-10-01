from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_package():
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Gathering all transfer package members for v3 ===")
    core_files = [
        "requirements.txt",
        "configs/constraints-py311.txt",
        "train.py",
        "evaluate.py",
        "inference.py",
        "validate_submission.py",
        "generate_submission.py",
        "configs/iteration3_resnet34.yaml",
        "configs/b0_resnet34.yaml",
        "scripts/cloud_bootstrap.sh",
        "scripts/iteration3_experiment.py",
        "scripts/iteration3_cloud_preflight.py",
        "artifacts/folds_manifest.json",
        "artifacts/partitions_migrated_v1.json",
        "artifacts/reports/mining_bank_v1.json",
        "artifacts/reports/mining_bank_v1.provenance.json",
        "artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt",
    ]

    # Add src tree
    for p in sorted((root / "src").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        core_files.append(rel)

    # Add tests tree
    for p in sorted((root / "tests").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        core_files.append(rel)

    # Add dataset files
    dataset_dir = root / "data" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    train_images = sorted((dataset_dir / "train" / "train_images").glob("*.jpeg"))
    test_images = sorted((dataset_dir / "test" / "test_images").glob("*.jpeg"))
    train_json = dataset_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    print(f"Found {len(core_files)} code/config/checkpoint files.")
    print(f"Found {len(train_images)} train images (expected 707).")
    print(f"Found {len(test_images)} test images (expected 180).")
    print(f"Train annotations file: {train_json.is_file()}")

    assert len(train_images) == 707, f"Expected 707 train images, got {len(train_images)}"
    assert len(test_images) == 180, f"Expected 180 test images, got {len(test_images)}"
    assert train_json.is_file(), "Train annotations missing"

    all_members = list(core_files)
    all_members.append(train_json.relative_to(root).as_posix())
    for img in train_images:
        all_members.append(img.relative_to(root).as_posix())
    for img in test_images:
        all_members.append(img.relative_to(root).as_posix())

    print(f"Total package members to hash and archive: {len(all_members)}")

    # Mechanically compute sha256 and size for every member
    members_manifest = []
    total_uncompressed_bytes = 0
    for idx, rel_path in enumerate(all_members):
        full_p = root / rel_path
        if not full_p.is_file():
            raise FileNotFoundError(f"Missing package member: {full_p}")
        size = full_p.stat().st_size
        total_uncompressed_bytes += size
        sha = compute_file_sha256(full_p)
        members_manifest.append({
            "path": rel_path,
            "sha256": sha,
            "size_bytes": size,
        })
        if (idx + 1) % 200 == 0 or (idx + 1) == len(all_members):
            print(f"  Processed {idx + 1}/{len(all_members)} members...")

    archive_path = root / "artifacts" / "iteration3_private_transfer_v3.tar.gz"
    print(f"\n=== Step 2: Creating tar.gz archive at {archive_path} ===")
    with tarfile.open(archive_path, "w:gz") as tar:
        for m in members_manifest:
            full_p = root / m["path"]
            tar.add(full_p, arcname=m["path"])

    archive_bytes = archive_path.stat().st_size
    archive_sha = compute_file_sha256(archive_path)
    print(f"Archive created successfully: {archive_bytes} bytes ({archive_bytes / (1024**2):.2f} MB)")
    print(f"Archive SHA256: {archive_sha}")

    # Build cloud_transfer_manifest_v3.json
    manifest_v3_data = {
        "manifest_version": "3.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_platform": "Linux x86_64 / PyTorch 2.6.0+cu124 on Ubuntu 22.04",
        "recommended_container_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "python_compatibility": "Tested locally on Python 3.12; container standard Python is 3.11 with pinned configs/constraints-py311.txt",
        "linux_verification_status": "PENDING_REMOTE_POD_EXECUTION",
        "production_run_mode": "FRESH_A_B_ONLY",
        "resume_status": "UNVERIFIED_AND_EXCLUDED",
        "archive": {
            "archive_filename": "iteration3_private_transfer_v3.tar.gz",
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_bytes,
            "archive_size_mb": round(archive_bytes / (1024**2), 2),
            "total_uncompressed_bytes": total_uncompressed_bytes,
            "total_uncompressed_mb": round(total_uncompressed_bytes / (1024**2), 2),
        },
        "dataset_archive_alternative": {
            "path": "data/filament-segmentation-2026.zip",
            "sha256": "56d8cd2927859fa7df72bd28ad1586f53c5814df6f81db02e755223e430657df",
            "size_bytes": 703574877,
            "train_image_count": 707,
            "test_image_count": 180,
        },
        "total_files": len(members_manifest),
        "members": members_manifest,
    }

    manifest_v3_path = reports_dir / "cloud_transfer_manifest_v3.json"
    with open(manifest_v3_path, "w", encoding="utf-8") as f:
        json.dump(manifest_v3_data, f, indent=2)
    print(f"Saved manifest v3 to {manifest_v3_path}")
    return archive_path, archive_sha, len(members_manifest)


if __name__ == "__main__":
    build_package()
