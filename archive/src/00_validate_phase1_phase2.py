import argparse
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS"
FEATURES_DIR = WORKSPACE_DIR / "data" / "02_frozen_features"
REPORTS_DIR = WORKSPACE_DIR / "reports"


def list_ids_from_images(images_dir: Path):
    ids = set()
    for p in images_dir.glob("*.nii.gz"):
        name = p.name
        if name.endswith("_0000.nii.gz"):
            ids.add(name[: -len("_0000.nii.gz")])
    return ids


def list_ids_from_labels(labels_dir: Path):
    ids = set()
    for p in labels_dir.glob("*.nii.gz"):
        ids.add(p.name[: -len(".nii.gz")])
    return ids


def approx_tuple_equal(a, b, tol=1e-5):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(float(x) - float(y)) > tol:
            return False
    return True


def check_phase1(sample_size: int = 5):
    result = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    images_tr = DATASET_DIR / "imagesTr"
    labels_tr = DATASET_DIR / "labelsTr"
    images_ts = DATASET_DIR / "imagesTs"
    labels_ts = DATASET_DIR / "labelsTs"
    dataset_json = DATASET_DIR / "dataset.json"

    required_paths = [images_tr, labels_tr, images_ts, labels_ts, dataset_json]
    for req in required_paths:
        if not req.exists():
            result["errors"].append(f"Missing required path: {req}")

    if result["errors"]:
        result["status"] = "fail"
        return result

    tr_image_ids = list_ids_from_images(images_tr)
    tr_label_ids = list_ids_from_labels(labels_tr)
    ts_image_ids = list_ids_from_images(images_ts)
    ts_label_ids = list_ids_from_labels(labels_ts)

    result["stats"]["train_images"] = len(tr_image_ids)
    result["stats"]["train_labels"] = len(tr_label_ids)
    result["stats"]["test_images"] = len(ts_image_ids)
    result["stats"]["test_labels"] = len(ts_label_ids)

    missing_tr_labels = sorted(tr_image_ids - tr_label_ids)
    missing_tr_images = sorted(tr_label_ids - tr_image_ids)
    missing_ts_labels = sorted(ts_image_ids - ts_label_ids)
    missing_ts_images = sorted(ts_label_ids - ts_image_ids)

    if missing_tr_labels:
        result["errors"].append(f"Train images missing labels: {len(missing_tr_labels)}")
    if missing_tr_images:
        result["errors"].append(f"Train labels missing images: {len(missing_tr_images)}")
    if missing_ts_labels:
        result["errors"].append(f"Test images missing labels: {len(missing_ts_labels)}")
    if missing_ts_images:
        result["errors"].append(f"Test labels missing images: {len(missing_ts_images)}")

    overlap = sorted(tr_image_ids & ts_image_ids)
    result["stats"]["train_test_overlap_count"] = len(overlap)
    if overlap:
        result["errors"].append(f"Train/test patient overlap found: {len(overlap)}")

    label_samples = []
    pooled = []
    pooled.extend(sorted(tr_label_ids)[:sample_size])
    pooled.extend(sorted(ts_label_ids)[:sample_size])

    binary_violations = 0
    geometry_mismatches = 0
    for pid in pooled:
        split = "train" if pid in tr_label_ids else "test"
        label_path = labels_tr / f"{pid}.nii.gz" if split == "train" else labels_ts / f"{pid}.nii.gz"
        image_path = images_tr / f"{pid}_0000.nii.gz" if split == "train" else images_ts / f"{pid}_0000.nii.gz"

        if not label_path.exists() or not image_path.exists():
            continue

        lbl_img = sitk.ReadImage(str(label_path))
        img = sitk.ReadImage(str(image_path))
        lbl_arr = sitk.GetArrayFromImage(lbl_img)
        unique_values = np.unique(lbl_arr)
        is_binary = np.all(np.isin(unique_values, [0, 1]))
        if not is_binary:
            binary_violations += 1

        geom_match = (
            tuple(lbl_img.GetSize()) == tuple(img.GetSize())
            and approx_tuple_equal(tuple(lbl_img.GetSpacing()), tuple(img.GetSpacing()))
            and approx_tuple_equal(tuple(lbl_img.GetOrigin()), tuple(img.GetOrigin()))
            and approx_tuple_equal(tuple(lbl_img.GetDirection()), tuple(img.GetDirection()))
        )
        if not geom_match:
            geometry_mismatches += 1

        label_samples.append(
            {
                "patient_id": pid,
                "split": split,
                "unique_values": unique_values.tolist(),
                "binary": bool(is_binary),
                "geometry_matches_image": bool(geom_match),
            }
        )

    result["stats"]["sampled_labels"] = len(label_samples)
    result["stats"]["binary_violations_in_samples"] = binary_violations
    result["stats"]["geometry_mismatches_in_samples"] = geometry_mismatches
    result["stats"]["label_samples"] = label_samples

    if binary_violations > 0:
        result["errors"].append(f"Found non-binary label values in {binary_violations} sampled masks")
    if geometry_mismatches > 0:
        result["errors"].append(f"Found geometry mismatches in {geometry_mismatches} sampled pairs")

    result["status"] = "fail" if result["errors"] else "pass"
    return result


def load_manifest_records(manifest_path: Path):
    records = []
    if not manifest_path.exists():
        return records
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_invalid_json_line": line})
    return records


def check_phase2(split_name: str, expected_ids: set, sample_size: int = 5):
    out_dir = FEATURES_DIR / split_name
    result = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "stats": {"split": split_name, "expected_patients": len(expected_ids)},
    }

    if not out_dir.exists():
        result["errors"].append(f"Missing split output directory: {out_dir}")
        result["status"] = "fail"
        return result

    missing_by_type = {
        "profound": [],
        "sam_embedding": [],
        "sam_logits": [],
        "meta": [],
    }

    for pid in sorted(expected_ids):
        required = {
            "profound": out_dir / f"{pid}_profound.npy",
            "sam_embedding": out_dir / f"{pid}_sam_embedding.npy",
            "sam_logits": out_dir / f"{pid}_sam_logits.npy",
            "meta": out_dir / f"{pid}_meta.json",
        }
        for key, path in required.items():
            if not path.exists():
                missing_by_type[key].append(pid)

    for key, ids in missing_by_type.items():
        result["stats"][f"missing_{key}"] = len(ids)
        if ids:
            result["errors"].append(f"Missing {key} files: {len(ids)}")

    manifest_path = out_dir / "manifest.jsonl"
    records = load_manifest_records(manifest_path)
    result["stats"]["manifest_exists"] = manifest_path.exists()
    result["stats"]["manifest_records"] = len(records)
    if not manifest_path.exists():
        result["errors"].append(f"Missing manifest file: {manifest_path}")

    record_ids = []
    invalid_manifest_lines = 0
    for rec in records:
        if "_invalid_json_line" in rec:
            invalid_manifest_lines += 1
            continue
        pid = rec.get("patient_id")
        if pid is not None:
            record_ids.append(pid)

    if invalid_manifest_lines > 0:
        result["errors"].append(f"Invalid manifest JSON lines: {invalid_manifest_lines}")

    duplicates = len(record_ids) - len(set(record_ids))
    result["stats"]["manifest_duplicate_ids"] = duplicates
    if duplicates > 0:
        result["errors"].append(f"Duplicate patient IDs in manifest: {duplicates}")

    sampled_ids = sorted(list(expected_ids))[:sample_size]
    samples = []
    nan_inf_violations = 0
    metadata_shape_violations = 0

    for pid in sampled_ids:
        profound_path = out_dir / f"{pid}_profound.npy"
        sam_emb_path = out_dir / f"{pid}_sam_embedding.npy"
        sam_logits_path = out_dir / f"{pid}_sam_logits.npy"
        meta_path = out_dir / f"{pid}_meta.json"

        if not (profound_path.exists() and sam_emb_path.exists() and sam_logits_path.exists() and meta_path.exists()):
            continue

        profound = np.load(profound_path)
        sam_emb = np.load(sam_emb_path)
        sam_logits = np.load(sam_logits_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        finite = bool(np.isfinite(profound).all() and np.isfinite(sam_emb).all() and np.isfinite(sam_logits).all())
        if not finite:
            nan_inf_violations += 1

        meta_shapes = meta.get("tensors", {})
        shape_ok = (
            list(profound.shape) == meta_shapes.get("profound_output_shape_cdhw", [])
            and list(sam_emb.shape) == meta_shapes.get("sam_embedding_shape_cdhw", [])
            and list(sam_logits.shape) == meta_shapes.get("sam_logits_shape_dhw", [])
        )
        if not shape_ok:
            metadata_shape_violations += 1

        samples.append(
            {
                "patient_id": pid,
                "profound_shape": list(profound.shape),
                "sam_embedding_shape": list(sam_emb.shape),
                "sam_logits_shape": list(sam_logits.shape),
                "finite": finite,
                "meta_shapes_match_arrays": shape_ok,
            }
        )

    result["stats"]["sampled_patients"] = len(samples)
    result["stats"]["tensor_nan_inf_violations"] = nan_inf_violations
    result["stats"]["metadata_shape_violations"] = metadata_shape_violations
    result["stats"]["samples"] = samples

    if nan_inf_violations > 0:
        result["errors"].append(f"Found NaN/inf values in {nan_inf_violations} sampled patients")
    if metadata_shape_violations > 0:
        result["errors"].append(f"Metadata shape mismatch in {metadata_shape_violations} sampled patients")

    result["status"] = "fail" if result["errors"] else "pass"
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 1 and Phase 2 implementation readiness.")
    parser.add_argument("--sample-size", type=int, default=5, help="How many patients to sample per split for deep checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "phase1_phase2_validation.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    images_tr = DATASET_DIR / "imagesTr"
    images_ts = DATASET_DIR / "imagesTs"
    expected_train_ids = list_ids_from_images(images_tr) if images_tr.exists() else set()
    expected_test_ids = list_ids_from_images(images_ts) if images_ts.exists() else set()

    report = {
        "phase1": check_phase1(sample_size=args.sample_size),
        "phase2_train": check_phase2("train", expected_train_ids, sample_size=args.sample_size),
        "phase2_test": check_phase2("test", expected_test_ids, sample_size=args.sample_size),
    }

    overall_fail = any(report[k]["status"] == "fail" for k in report)
    report["overall_status"] = "fail" if overall_fail else "pass"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Validation report written to: {args.output}")
    print(f"Overall status: {report['overall_status']}")
    for key in ["phase1", "phase2_train", "phase2_test"]:
        print(f"- {key}: {report[key]['status']} (errors={len(report[key]['errors'])})")


if __name__ == "__main__":
    main()
