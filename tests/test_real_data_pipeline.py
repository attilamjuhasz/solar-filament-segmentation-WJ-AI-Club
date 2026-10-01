import os
import tempfile
import cv2
import numpy as np
import pytest

from src.contracts import NATIVE_IMAGE_SHAPE
from src.data.annotations import load_coco_annotations
from src.data.targets import make_targets
from src.evaluation.competition_adapter import evaluate_entry_pq, evaluate_dataset_pq
from src.inference.instances import extract_instances_from_maps
from src.inference.rle import (
    decode_instance,
    encode_instance,
    validate_submission_csv,
    write_submission_csv,
)

DATA_ROOT = r"C:\Users\Xxran\Downloads\solar filament\data\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
TRAIN_IMAGES = os.path.join(DATA_ROOT, "train", "train_images")


@pytest.mark.skipif(not os.path.exists(TRAIN_JSON), reason="Competition dataset not downloaded")
def test_end_to_end_real_observation():
    # 1. Load annotation index (metadata loaded in ~1s, masks decoded lazily)
    print("\n[Step 1] Loading real training annotations index...")
    index = load_coco_annotations(TRAIN_JSON)
    assert len(index) > 0, "No entries found in annotation index"
    print(f"Loaded index containing {len(index)} observation entries.")

    # 2. Pick a real observation with multiple instances
    target_entry = None
    for entry in index.by_annotator_image.values():
        if len(entry.instances) >= 2:
            target_entry = entry
            break

    assert target_entry is not None, "Could not find an observation with >= 2 instances"
    obs_id = target_entry.observation_id
    img_path = os.path.join(TRAIN_IMAGES, target_entry.file_name)
    assert os.path.exists(img_path), f"Image file not found: {img_path}"

    # 3. Load actual image with OpenCV
    print(f"[Step 2] Loading real image: {target_entry.file_name} ...")
    img = cv2.imread(img_path)
    assert img is not None, "Failed to load real solar image"
    assert img.shape[:2] == NATIVE_IMAGE_SHAPE

    # Decode only the masks for this observation (e.g. 2-5 masks = ~8-20 MB total)
    gt_masks = [inst.mask for inst in target_entry.instances]
    print(f"Decoded {len(gt_masks)} ground-truth filaments on-demand. Areas: {[m.sum() for m in gt_masks]}")

    # 4. Generate multi-task supervision bundle
    print("[Step 3] Generating multi-task target bundle (boundaries, center heatmap, offsets)...")
    bundle = make_targets(gt_masks, shape=NATIVE_IMAGE_SHAPE)
    assert bundle.foreground_mask.shape == NATIVE_IMAGE_SHAPE
    assert bundle.boundary_mask.shape == NATIVE_IMAGE_SHAPE
    assert bundle.center_heatmap.shape == NATIVE_IMAGE_SHAPE
    assert bundle.offset_field.shape == (2, 2048, 2048)

    # 5. Extract predicted instances from the generated maps
    print("[Step 4] Running marker-controlled watershed instance extraction...")
    blurred_fg = cv2.GaussianBlur(bundle.foreground_mask.astype(np.float32), (5, 5), 1.0)
    blurred_center = cv2.GaussianBlur(bundle.center_heatmap, (5, 5), 1.0)
    boundary_map = bundle.boundary_mask.astype(np.float32)

    instances = extract_instances_from_maps(
        foreground_prob=blurred_fg,
        center_prob=blurred_center,
        boundary_prob=boundary_map,
        obs_id=obs_id,
        high_threshold=0.40,
        low_threshold=0.20,
        min_area=16
    )

    assert len(instances) >= 1, "Failed to extract instances from real observation"
    pred_masks = [decode_instance(inst.rle_counts) for inst in instances]
    print(f"Extracted {len(instances)} predicted instances.")

    # 6. Evaluate strict Panoptic Quality against real ground truth
    print("[Step 5] Evaluating Panoptic Quality (strict IoU > 0.5 parity)...")
    total_iou, tp, fp, fn, o2m, m2o = evaluate_entry_pq(gt_masks, pred_masks)
    print(f"Result: TP={tp}, FP={fp}, FN={fn}, Total IoU={total_iou:.3f}")
    assert tp >= 1, "Expected at least 1 true positive match on ground-truth targets"

    dataset_stats = evaluate_dataset_pq([(gt_masks, pred_masks)])
    assert dataset_stats.pq > 0.0
    print(f"Panoptic Quality: PQ={dataset_stats.pq:.4f}, SQ={dataset_stats.sq:.4f}, RQ={dataset_stats.rq:.4f}")

    # 7. Write to submission.csv and validate
    print("[Step 6] Writing to submission.csv and running strict format validator...")
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_csv = os.path.join(tmpdir, "submission.csv")
        rows = [(inst.filament_id, inst.rle_counts) for inst in instances]
        write_submission_csv(sub_csv, rows)

        report = validate_submission_csv(
            sub_csv,
            expected_observation_ids={obs_id},
            verify_rle_decoding=True
        )
        assert report["is_valid"] is True
        assert report["total_instances"] == len(instances)
        print(f"SUCCESS: submission.csv validated with {len(instances)} instances for {obs_id}!")
