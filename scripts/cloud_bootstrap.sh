#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Solar Filament Segmentation - Iteration 3 Cloud Bootstrap & Execution Script
# Recommended Container: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Target GPU: 1x RTX 4090 / A5000 / A4000 (>= 16GB VRAM recommended)
# ==============================================================================

MODE="preflight"
DEADLINE_TIMESTAMP="${DEADLINE_TIMESTAMP:-}"
CLEANUP_MARGIN_SECONDS=60
PIP_BIN="${PIP_BIN:-pip}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TAR_BIN="${TAR_BIN:-tar}"
SHA256_BIN="${SHA256_BIN:-sha256sum}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train|train)
            MODE="train"
            shift
            ;;
        --preflight|preflight)
            MODE="preflight"
            shift
            ;;
        --deadline-timestamp)
            DEADLINE_TIMESTAMP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--preflight | --train --deadline-timestamp <EPOCH_SECONDS>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--preflight | --train --deadline-timestamp <EPOCH_SECONDS>]"
            exit 1
            ;;
    esac
done

echo "=== [Phase 1] Archive Extraction & Integrity Verification ==="
if [ -f "iteration3_private_transfer_v3.tar.gz" ]; then
    echo "Extracting iteration3_private_transfer_v3.tar.gz..."
    "$TAR_BIN" -xzf iteration3_private_transfer_v3.tar.gz
elif [ -f "iteration3_private_transfer_v2.tar.gz" ]; then
    echo "Extracting iteration3_private_transfer_v2.tar.gz..."
    "$TAR_BIN" -xzf iteration3_private_transfer_v2.tar.gz
elif [ -f "data/filament-segmentation-2026.zip" ]; then
    echo "Extracting raw dataset zip..."
    "$PYTHON_BIN" -c "import zipfile; zipfile.ZipFile('data/filament-segmentation-2026.zip').extractall('data/filament-segmentation-2026')"
else
    echo "Dataset images already present or extracted."
fi

echo "=== [Phase 2] Pinned Environment Installation ==="
# Install PyTorch 2.6.0 + CUDA 12.4 wheels from official index, followed by pinned constraints
"$PIP_BIN" install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if [ -f "configs/constraints-py311.txt" ]; then
    echo "Installing requirements with pinned Python 3.11 constraints (configs/constraints-py311.txt)..."
    "$PIP_BIN" install --no-cache-dir -c configs/constraints-py311.txt -r requirements.txt
else
    "$PIP_BIN" install --no-cache-dir -r requirements.txt
fi

echo "=== [Phase 3] Import-Only Verification ==="
"$PYTHON_BIN" -c "
import torch, torchvision, numpy, pycocotools, yaml, scipy, pandas, sklearn, skimage, PIL
print(f'Imports verified successfully. PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"

echo "=== [Phase 4] Read-Only Cloud Preflight & Hardware Audit ==="
"$PYTHON_BIN" scripts/iteration3_cloud_preflight.py

echo "=== [Phase 5] Short Disposable Benchmark (3 steps) ==="
"$PYTHON_BIN" scripts/iteration3_cloud_preflight.py --benchmark

if [ "$MODE" != "train" ]; then
    echo ""
    echo "=========================================================================="
    echo "=== [GO/NO-GO CHECKPOINT] Setup, Preflight, and Benchmark Complete ==="
    echo "Default bootstrap execution halts here by design for Codex go/no-go review."
    echo "Full training was NOT started."
    echo ""
    echo "To authorize and run full training under caller-supplied deadline, execute:"
    echo "  ./scripts/cloud_bootstrap.sh --train --deadline-timestamp <EPOCH_SECONDS>"
    echo "=========================================================================="
    exit 0
fi

echo "=== [Phase 6] Validating Caller-Supplied Training Deadline ==="
if [ -z "$DEADLINE_TIMESTAMP" ]; then
    echo "ERROR: Training mode requires --deadline-timestamp <EPOCH_SECONDS> (or DEADLINE_TIMESTAMP env var)." >&2
    exit 1
fi

if ! [[ "$DEADLINE_TIMESTAMP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --deadline-timestamp must be an integer unix epoch timestamp, got: '${DEADLINE_TIMESTAMP}'" >&2
    exit 1
fi

CURRENT_TIME=$("$PYTHON_BIN" -c "import time; print(int(time.time()))")
REMAINING_BUDGET=$((DEADLINE_TIMESTAMP - CURRENT_TIME))

if [ "$REMAINING_BUDGET" -le "$CLEANUP_MARGIN_SECONDS" ]; then
    echo "ERROR: Provided deadline ${DEADLINE_TIMESTAMP} has expired or does not leave adequate cleanup margin (current: ${CURRENT_TIME}, margin: ${CLEANUP_MARGIN_SECONDS}s, remaining: ${REMAINING_BUDGET}s)." >&2
    exit 1
fi

EFFECTIVE_BUDGET=$((REMAINING_BUDGET - CLEANUP_MARGIN_SECONDS))
echo "Caller-supplied absolute deadline: ${DEADLINE_TIMESTAMP} (current: ${CURRENT_TIME}, remaining budget: ${EFFECTIVE_BUDGET}s, cleanup margin: ${CLEANUP_MARGIN_SECONDS}s)"

echo "=== [Phase 7] Sequential Iteration 3 Full Execution (Arm A -> Arm B -> Tuning -> Comparison) ==="
"$PYTHON_BIN" scripts/iteration3_experiment.py \
    --budget-seconds "${EFFECTIVE_BUDGET}" \
    --deadline-timestamp "${DEADLINE_TIMESTAMP}"

echo "=== [Phase 8] Export Iteration 3 Artifacts ==="
EXPORT_NAME="iteration3_results_$(date +%Y%m%d_%H%M%S).tar.gz"
"$TAR_BIN" -czf "${EXPORT_NAME}" \
    artifacts/reports/ \
    artifacts/runs/ \
    $(ls artifacts/submission_candidate_3* 2>/dev/null || true)

echo "Export completed: ${EXPORT_NAME}"
"$SHA256_BIN" "${EXPORT_NAME}"
echo "Ready for Codex independent review and billing termination."
