from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def generate_v3_reports():
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "artifacts" / "reports"

    archive_path = root / "artifacts" / "iteration3_private_transfer_v3.tar.gz"
    archive_sha = compute_file_sha256(archive_path)
    archive_size = archive_path.stat().st_size

    parent_path = root / "artifacts" / "runs" / "run_20260930_111126_15cb60" / "checkpoints" / "epoch_006.pt"
    folds_path = root / "artifacts" / "folds_manifest.json"
    parts_path = root / "artifacts" / "partitions_migrated_v1.json"
    bank_path = root / "artifacts" / "reports" / "mining_bank_v1.json"
    prov_path = root / "artifacts" / "reports" / "mining_bank_v1.provenance.json"
    req_path = root / "requirements.txt"
    const_path = root / "configs" / "constraints-py311.txt"
    train_path = root / "train.py"
    exp_path = root / "scripts" / "iteration3_experiment.py"
    boot_path = root / "scripts" / "cloud_bootstrap.sh"
    man_v3_path = reports_dir / "cloud_transfer_manifest_v3.json"

    parent_sha = compute_file_sha256(parent_path)
    folds_sha = compute_file_sha256(folds_path)
    parts_sha = compute_file_sha256(parts_path)
    bank_sha = compute_file_sha256(bank_path)
    prov_sha = compute_file_sha256(prov_path)
    req_sha = compute_file_sha256(req_path)
    const_sha = compute_file_sha256(const_path)
    train_sha = compute_file_sha256(train_path)
    exp_sha = compute_file_sha256(exp_path)
    boot_sha = compute_file_sha256(boot_path)
    man_v3_sha = compute_file_sha256(man_v3_path)

    v3_report = {
        "report_version": "3.0.0",
        "milestone": "Iteration 3 Fresh-Training Handoff & Cloud Readiness Pass v3",
        "status": "ready_for_codex_cloud_execution",
        "production_run_mode": "FRESH_A_B_ONLY",
        "resume_status": "UNVERIFIED_AND_EXCLUDED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "handoff_summary": {
            "archive_path": "artifacts/iteration3_private_transfer_v3.tar.gz",
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_size,
            "archive_size_mb": round(archive_size / (1024**2), 2),
            "manifest_path": "artifacts/reports/cloud_transfer_manifest_v3.json",
            "manifest_sha256": man_v3_sha,
            "total_packaged_members": 942,
            "train_images_count": 707,
            "test_images_count": 180,
            "pinned_cloud_constraints": "configs/constraints-py311.txt",
        },
        "cloud_bootstrap_contract": {
            "default_action": "Executes phases 1-5 (archive unpack, pinned install, import check, preflight, benchmark) and halts with exit code 0 for Codex go/no-go review",
            "full_training_launch": "Codex runs iteration3_experiment.py directly with caller-supplied inherited deadline, leaving >= 15m export margin, or via bootstrap --train with explicit --deadline-timestamp",
            "deadline_safety": "Rejects missing or expired deadlines before any GPU compute",
        },
        "immutable_input_hashes": {
            "parent_checkpoint_file": parent_sha,
            "parent_model_state": "f7d19980cd2ed382ca4683b6b0932ae404e98359976b5bd9eaebbc44bb4f202c",
            "folds_manifest": folds_sha,
            "partitions": parts_sha,
            "mining_bank": bank_sha,
            "mining_provenance": prov_sha,
            "dataset_zip": "56d8cd2927859fa7df72bd28ad1586f53c5814df6f81db02e755223e430657df",
        },
        "source_code_hashes": {
            "cloud_bootstrap_sh": boot_sha,
            "iteration3_experiment_py": exp_sha,
            "train_py": train_sha,
            "requirements_txt": req_sha,
            "constraints_py311_txt": const_sha,
            "cloud_transfer_manifest_v3_json": man_v3_sha,
        },
        "test_results": {
            "iteration3_contracts": "13 passed, 2 skipped (optional resume tests excluded)",
            "full_test_suite": "91 passed, 2 skipped, 0 failed in 46.32s",
        },
        "unresolved_blockers": "None on source or package readiness. Remote Linux container execution pending launch by Codex.",
    }

    v3_report_path = reports_dir / "iteration3_readiness_report_v3.json"
    with open(v3_report_path, "w", encoding="utf-8") as f:
        json.dump(v3_report, f, indent=2)
    print(f"Saved v3 readiness report to {v3_report_path}")

    # Update work_state.json
    work_state_path = reports_dir / "work_state.json"
    work_state_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "Iteration 3 Handoff v3 Complete",
        "status": "ready_for_codex_cloud_execution",
        "production_run_mode": "FRESH_A_B_ONLY",
        "resume_status": "UNVERIFIED_AND_EXCLUDED",
        "archive": {
            "path": "artifacts/iteration3_private_transfer_v3.tar.gz",
            "sha256": archive_sha,
            "size_mb": round(archive_size / (1024**2), 2),
            "members": 942,
        },
        "manifest": {
            "path": "artifacts/reports/cloud_transfer_manifest_v3.json",
            "sha256": man_v3_sha,
        },
        "readiness_report_v3": {
            "path": "artifacts/reports/iteration3_readiness_report_v3.json",
        },
        "tests": {
            "contracts_passed": 13,
            "contracts_skipped": 2,
            "full_suite_passed": 91,
            "full_suite_skipped": 2,
            "full_suite_failed": 0,
        },
        "budget_spent_usd": 0.00,
        "remaining_budget_usd": 1.00,
        "blockers": "None locally; Linux container verification pending remote pod launch by Codex",
    }
    with open(work_state_path, "w", encoding="utf-8") as f:
        json.dump(work_state_data, f, indent=2)
    print(f"Updated work state in {work_state_path}")


if __name__ == "__main__":
    generate_v3_reports()
