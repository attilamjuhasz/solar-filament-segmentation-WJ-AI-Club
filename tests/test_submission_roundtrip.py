import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
from pycocotools import mask as coco_mask

from src.contracts import NATIVE_IMAGE_SHAPE
from src.inference.rle import (
    decode_instance,
    encode_instance,
    parse_compressed_rle_runs,
    validate_compressed_rle,
    validate_submission_csv,
    write_submission_csv,
)


def test_encode_and_decode_roundtrip():
    # Test 1: Single pixel
    m1 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m1[100, 200] = 1
    rle1 = encode_instance(m1)
    dec1 = decode_instance(rle1)
    assert np.array_equal(m1, dec1)

    # Test 2: Thin 1-pixel spine
    m2 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m2[1024, 100:1900] = 1
    m2[1000:1025, 500] = 1  # barb
    rle2 = encode_instance(m2)
    dec2 = decode_instance(rle2)
    assert np.array_equal(m2, dec2)

    # Test 3: Disjoint multi-component instance
    m3 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    m3[200:250, 300:350] = 1
    m3[800:850, 900:950] = 1
    rle3 = encode_instance(m3)
    dec3 = decode_instance(rle3)
    assert np.array_equal(m3, dec3)


def test_encode_rejects_invalid_inputs():
    # Empty mask
    empty = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
    with pytest.raises(ValueError, match="empty mask"):
        encode_instance(empty)

    # Wrong shape (e.g. 512x512 from naive baseline)
    wrong_shape = np.ones((512, 512), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected shape"):
        encode_instance(wrong_shape)

    # Non-binary
    non_binary = np.full(NATIVE_IMAGE_SHAPE, 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        encode_instance(non_binary)


def test_512x512_rle_rejection_regression():
    """Codex P1 finding #2: Verify 512x512 RLE decoded as 2048x2048 is rejected BEFORE pycocotools distorts it."""
    m512 = np.zeros((512, 512), dtype=np.uint8, order="F")
    m512[100:120, 100:120] = 1
    encoded512 = coco_mask.encode(m512)
    rle_512_str = encoded512["counts"].decode("ascii")

    # 1. Direct validation must detect pixel count mismatch (262,144 vs 4,194,304)
    with pytest.raises(ValueError, match="total pixel count 262144 does not match expected shape"):
        validate_compressed_rle(rle_512_str, NATIVE_IMAGE_SHAPE)

    # 2. decode_instance must reject it
    with pytest.raises(ValueError, match="total pixel count 262144"):
        decode_instance(rle_512_str, NATIVE_IMAGE_SHAPE)

    # 3. validate_submission_csv must reject a CSV containing it
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "bad_submission.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filament_id,segmentation_rle\n")
            f.write(f"20110120105534Ch_1,{rle_512_str}\n")
        
        with pytest.raises(ValueError, match="total pixel count 262144"):
            validate_submission_csv(csv_path)


def test_strict_unquoted_csv_and_constraints():
    """Verify that quotes are strictly forbidden and expected_observation_ids constraint is enforced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mask = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        mask[500:550, 600:650] = 1
        valid_rle = encode_instance(mask)

        # 1. Quoted lines must fail
        quoted_csv = os.path.join(tmpdir, "quoted.csv")
        with open(quoted_csv, "w", encoding="utf-8") as f:
            f.write("filament_id,segmentation_rle\n")
            f.write(f'"20110120105534Ch_1","{valid_rle}"\n')
        
        with pytest.raises(ValueError, match="contains quote characters"):
            validate_submission_csv(quoted_csv)

        # 2. Empty expected_observation_ids set constraint must fail for any row
        valid_csv = os.path.join(tmpdir, "valid.csv")
        with open(valid_csv, "w", encoding="utf-8") as f:
            f.write("filament_id,segmentation_rle\n")
            f.write(f"20110120105534Ch_1,{valid_rle}\n")

        with pytest.raises(ValueError, match="not in expected test set"):
            validate_submission_csv(valid_csv, expected_observation_ids=set())

        # 3. Successful validation when obs_id is in expected_observation_ids
        report = validate_submission_csv(valid_csv, expected_observation_ids={"20110120105534Ch"})
        assert report["is_valid"] is True
        assert report["total_instances"] == 1


def test_submission_csv_writing_and_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "submission.csv")

        mask1 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        mask1[500:550, 600:650] = 1
        rle1 = encode_instance(mask1)

        mask2 = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        mask2[1200:1220, 1300:1350] = 1
        rle2 = encode_instance(mask2)

        rows = [
            ("20110120105534Ch_1", rle1),
            ("20110120105534Ch_2", rle2),
            ("20110130110334Ch_1", rle1),
        ]

        count = write_submission_csv(csv_path, rows)
        assert count == 3

        report = validate_submission_csv(
            csv_path,
            expected_observation_ids={"20110120105534Ch", "20110130110334Ch"},
            verify_rle_decoding=True
        )
        assert report["is_valid"] is True
        assert report["total_instances"] == 3
        assert report["unique_observations"] == 2


def test_rle_alphabet_and_truncation_rejections():
    """Codex follow-up finding #9: Validate 6-bit alphabet (48..111) and truncated LEB integer rejections."""
    # 'p' has ASCII 112 (outside 48..111), should be rejected
    with pytest.raises(ValueError, match="valid alphabet is ASCII 48"):
        parse_compressed_rle_runs("p1o1")

    # Character < 48 (e.g. '+', ASCII 43)
    with pytest.raises(ValueError, match="valid alphabet is ASCII 48"):
        parse_compressed_rle_runs("+1o1")

    # Truncated continuation bit (e.g. 'P' has continuation bit set, but ends abruptly)
    with pytest.raises(ValueError, match="Unexpected end of compressed RLE string"):
        parse_compressed_rle_runs("P")


def test_header_only_csv_is_structurally_valid():
    """Feedback #2 [P2] #5: Header-only CSV is structurally valid (legitimate all-abstained inference)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "header_only.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filament_id,segmentation_rle\n")

        # validate_submission_csv accepts header-only as structurally valid
        report = validate_submission_csv(csv_path, expected_observation_ids={"20110120105534Ch"})
        assert report["is_valid"] is True
        assert report["total_instances"] == 0
        assert report["unique_observations"] == 0


def test_audit_submission_and_manifest_scenarios():
    """Feedback #2 [P2] #4 and #5: Comprehensive audit test suite for submission CSV and completion manifest.
    
    Covers:
    - positive instances
    - some abstained
    - all abstained (0 instances total)
    - failed status rejection
    - missing / duplicate observation records
    - mismatched counts
    - stale SHA256 binding
    - empty expected set
    """
    import json
    from src.inference.rle import audit_submission_and_manifest, compute_file_sha256

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Positive + Some Abstained valid scenario
        # obs1 has 2 instances, obs2 is abstained (0 instances)
        csv_path = os.path.join(tmpdir, "valid.csv")
        mask_native = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        mask_native[100:120, 100:120] = 1
        rle = encode_instance(mask_native, expected_shape=NATIVE_IMAGE_SHAPE)
        rows = [
            ("20110120105534Ch_1", rle),
            ("20110120105534Ch_2", rle),
        ]
        write_submission_csv(csv_path, rows)
        csv_sha = compute_file_sha256(csv_path)

        manifest_path = os.path.join(tmpdir, "valid.manifest.json")
        manifest = {
            "submission_csv": csv_path,
            "csv_sha256": csv_sha,
            "total_instances": 2,
            "observations": [
                {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
            ]
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        audit = audit_submission_and_manifest(
            csv_path, manifest_path,
            expected_observation_ids={"20110120105534Ch", "20110130110334Ch"}
        )
        assert audit["is_valid"] is True
        assert audit["processed_observations"] == 1
        assert audit["abstained_observations"] == 1
        assert audit["total_instances"] == 2

        # 2. All-abstained valid scenario (0 rows, header-only CSV)
        all_abstained_csv = os.path.join(tmpdir, "all_abstained.csv")
        with open(all_abstained_csv, "w", encoding="utf-8") as f:
            f.write("filament_id,segmentation_rle\n")
        all_abs_sha = compute_file_sha256(all_abstained_csv)

        all_abstained_manifest = os.path.join(tmpdir, "all_abstained.manifest.json")
        manifest_all_abs = {
            "submission_csv": all_abstained_csv,
            "csv_sha256": all_abs_sha,
            "total_instances": 0,
            "observations": [
                {"observation_id": "20110120105534Ch", "status": "abstained", "instance_count": 0},
                {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
            ]
        }
        with open(all_abstained_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_all_abs, f)

        audit_abs = audit_submission_and_manifest(
            all_abstained_csv, all_abstained_manifest,
            expected_observation_ids={"20110120105534Ch", "20110130110334Ch"}
        )
        assert audit_abs["is_valid"] is True
        assert audit_abs["total_instances"] == 0
        assert audit_abs["abstained_observations"] == 2

        # 3. Failed terminal status rejection
        manifest_failed = os.path.join(tmpdir, "failed.manifest.json")
        manifest_f = {
            "submission_csv": csv_path,
            "csv_sha256": csv_sha,
            "total_instances": 2,
            "observations": [
                {"observation_id": "20110120105534Ch", "status": "failed", "instance_count": 2},
                {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
            ]
        }
        with open(manifest_failed, "w", encoding="utf-8") as f:
            json.dump(manifest_f, f)
        with pytest.raises(ValueError, match="invalid terminal status 'failed'"):
            audit_submission_and_manifest(csv_path, manifest_failed)

        # 4. Duplicate observation records
        manifest_dup = os.path.join(tmpdir, "dup.manifest.json")
        manifest_d = {
            "submission_csv": csv_path,
            "csv_sha256": csv_sha,
            "total_instances": 2,
            "observations": [
                {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                {"observation_id": "20110120105534Ch", "status": "abstained", "instance_count": 0},
            ]
        }
        with open(manifest_dup, "w", encoding="utf-8") as f:
            json.dump(manifest_d, f)
        with pytest.raises(ValueError, match="Duplicate observation_id"):
            audit_submission_and_manifest(csv_path, manifest_dup)

        # 5. Mismatched instance counts (claims 5, CSV has 2)
        manifest_mismatch = os.path.join(tmpdir, "mismatch.manifest.json")
        manifest_m = {
            "submission_csv": csv_path,
            "csv_sha256": csv_sha,
            "total_instances": 5,
            "observations": [
                {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 5},
                {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
            ]
        }
        with open(manifest_mismatch, "w", encoding="utf-8") as f:
            json.dump(manifest_m, f)
        with pytest.raises(ValueError, match="Instance count mismatch"):
            audit_submission_and_manifest(csv_path, manifest_mismatch)

        # 6. Stale SHA256 binding
        manifest_stale = os.path.join(tmpdir, "stale.manifest.json")
        manifest_s = {
            "submission_csv": csv_path,
            "csv_sha256": "0" * 64,  # Fake checksum
            "total_instances": 2,
            "observations": [
                {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
            ]
        }
        with open(manifest_stale, "w", encoding="utf-8") as f:
            json.dump(manifest_s, f)
        with pytest.raises(ValueError, match="Stale manifest binding"):
            audit_submission_and_manifest(csv_path, manifest_stale)

        # 7. Empty expected set
        empty_csv = os.path.join(tmpdir, "empty.csv")
        with open(empty_csv, "w", encoding="utf-8") as f:
            f.write("filament_id,segmentation_rle\n")
        empty_sha = compute_file_sha256(empty_csv)
        empty_manifest = os.path.join(tmpdir, "empty.manifest.json")
        with open(empty_manifest, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": empty_csv,
                "csv_sha256": empty_sha,
                "total_instances": 0,
                "observations": []
            }, f)
        audit_empty = audit_submission_and_manifest(
            empty_csv, empty_manifest,
            expected_observation_ids=set()
        )
        assert audit_empty["is_valid"] is True
        assert audit_empty["total_manifest_observations"] == 0

        # 8. Feedback #3 [P2] #1: Missing or null csv_sha256 must fail
        manifest_no_sha = os.path.join(tmpdir, "no_sha.manifest.json")
        with open(manifest_no_sha, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": csv_path,
                "total_instances": 2,
                "observations": [
                    {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                    {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
                ]
            }, f)
        with pytest.raises(ValueError, match="missing required 'csv_sha256' field"):
            audit_submission_and_manifest(csv_path, manifest_no_sha)

        manifest_null_sha = os.path.join(tmpdir, "null_sha.manifest.json")
        with open(manifest_null_sha, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": csv_path,
                "csv_sha256": None,
                "total_instances": 2,
                "observations": [
                    {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                    {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
                ]
            }, f)
        with pytest.raises(ValueError, match="missing required 'csv_sha256' field"):
            audit_submission_and_manifest(csv_path, manifest_null_sha)

        manifest_bad_sha = os.path.join(tmpdir, "bad_sha.manifest.json")
        with open(manifest_bad_sha, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": csv_path,
                "csv_sha256": "not-a-valid-hex",
                "total_instances": 2,
                "observations": [
                    {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                    {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
                ]
            }, f)
        with pytest.raises(ValueError, match="valid 64-character lowercase hex"):
            audit_submission_and_manifest(csv_path, manifest_bad_sha)

        # 9. Feedback #3 [P2] #1: Boolean count rejection (False/True)
        manifest_bool_cnt = os.path.join(tmpdir, "bool_cnt.manifest.json")
        with open(manifest_bool_cnt, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": all_abstained_csv,
                "csv_sha256": all_abs_sha,
                "total_instances": 0,
                "observations": [
                    {"observation_id": "20110120105534Ch", "status": "abstained", "instance_count": False},
                    {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
                ]
            }, f)
        with pytest.raises(ValueError, match="invalid instance_count: False; must be non-negative integer"):
            audit_submission_and_manifest(all_abstained_csv, manifest_bool_cnt)

        # 10. Feedback #3 [P2] #1: Mismatched / malformed aggregate summary fields
        manifest_bad_summary = os.path.join(tmpdir, "bad_summary.manifest.json")
        with open(manifest_bad_summary, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": csv_path,
                "csv_sha256": csv_sha,
                "total_instances": 2,
                "total_test_observations": 999,  # Mismatch (actual is 2)
                "observations": [
                    {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                    {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
                ]
            }, f)
        with pytest.raises(ValueError, match="Manifest summary field 'total_test_observations' mismatch"):
            audit_submission_and_manifest(csv_path, manifest_bad_summary)

        manifest_negative_summary = os.path.join(tmpdir, "negative_summary.manifest.json")
        with open(manifest_negative_summary, "w", encoding="utf-8") as f:
            json.dump({
                "submission_csv": csv_path,
                "csv_sha256": csv_sha,
                "total_instances": 2,
                "abstained_observations_count": -1,  # Negative
                "observations": [
                    {"observation_id": "20110120105534Ch", "status": "processed", "instance_count": 2},
                    {"observation_id": "20110130110334Ch", "status": "abstained", "instance_count": 0},
                ]
            }, f)
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            audit_submission_and_manifest(csv_path, manifest_negative_summary)


def test_generate_submission_integration_smoke(monkeypatch):
    """Feedback #2 [P1] #2: Smoke test that generate_submission actually writes and reads CSV + manifest without NameError or schema issues."""
    import generate_submission as gen_mod
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_images")
        os.makedirs(test_dir, exist_ok=True)

        # Create dummy 2048x2048 JPEG with valid observation filename
        dummy_obs = "20150125172714Mh"
        img_path = os.path.join(test_dir, f"{dummy_obs}.jpeg")
        dummy_img = Image.new("L", (2048, 2048), color=50)
        dummy_img.save(img_path, format="JPEG")

        out_csv = os.path.join(tmpdir, "smoke_submission.csv")

        # Mock extract_filaments_from_image to return deterministic filament instances
        mask_native = np.zeros(NATIVE_IMAGE_SHAPE, dtype=np.uint8)
        mask_native[100:120, 100:120] = 1
        rle = encode_instance(mask_native, expected_shape=NATIVE_IMAGE_SHAPE)
        monkeypatch.setattr(
            gen_mod,
            "extract_filaments_from_image",
            lambda **kwargs: [(f"{dummy_obs}_1", rle)]
        )

        gen_mod.generate_submission(test_dir=test_dir, output_csv=out_csv, max_candidates=5)

        # Verify files exist on disk
        assert os.path.exists(out_csv)
        manifest_path = Path(out_csv).with_suffix(".manifest.json")
        assert os.path.exists(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_content = json.load(f)

        assert manifest_content["total_instances"] == 1
        assert manifest_content["detected_observations_count"] == 1
        assert manifest_content["observations"][0]["status"] == "processed"
        assert manifest_content["observations"][0]["instance_count"] == 1



