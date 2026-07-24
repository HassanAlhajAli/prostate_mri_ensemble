import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = WORKSPACE_DIR / "data" / "02_frozen_features" / "test"
PHASE4_DIR = WORKSPACE_DIR / "data" / "03_phase4_logits" / "test"
LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTs"
OUTPUT_DIR = WORKSPACE_DIR / "data" / "04_phase5_native_predictions"
REPORT_DIR = WORKSPACE_DIR / "reports"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def list_patient_ids(cache_dir: Path) -> List[str]:
    return sorted({p.name[: -len("_meta.json")] for p in cache_dir.glob("*_meta.json")})


def load_logits(path: Path) -> np.ndarray:
    logits = np.load(path).astype(np.float32)
    if logits.ndim == 3:
        return logits
    if logits.ndim == 4 and logits.shape[0] == 1:
        return logits[0]
    raise ValueError(f"Unexpected logits shape at {path}: {logits.shape}")


def resize_logits_to_shape(logits: np.ndarray, target_shape_dhw: List[int]) -> np.ndarray:
    tensor = torch.from_numpy(logits).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=tuple(int(x) for x in target_shape_dhw),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def dice_from_binary_masks(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return float((2.0 * intersection + eps) / (denom + eps))


def load_label_array(patient_id: str) -> Tuple[np.ndarray, sitk.Image]:
    label_path = LABEL_DIR / f"{patient_id}.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    label_img = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_img).astype(np.float32)
    label_arr = (label_arr > 0).astype(np.float32)
    return label_arr, label_img


def write_native_prediction(
    pred_mask: np.ndarray,
    reference_image: sitk.Image,
    output_path: Path,
) -> None:
    pred_img = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    pred_img.CopyInformation(reference_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pred_img, str(output_path))


def process_patient(patient_id: str, method_dirs: Dict[str, Path], output_root: Path) -> Dict[str, object]:
    meta_path = FEATURES_DIR / f"{patient_id}_meta.json"
    meta = load_json(meta_path)
    native_shape = meta.get("native_image", {}).get("shape_dhw")
    if not native_shape:
        raise ValueError(f"Missing native shape in metadata for {patient_id}")

    label_arr, label_img = load_label_array(patient_id)
    if list(label_arr.shape) != list(native_shape):
        raise ValueError(
            f"Label shape mismatch for {patient_id}: label={list(label_arr.shape)} meta={native_shape}"
        )

    result: Dict[str, object] = {
        "patient_id": patient_id,
        "native_shape": list(native_shape),
        "methods": {},
    }

    for method_name, method_dir in method_dirs.items():
        logits_path = method_dir / f"{patient_id}_logits.npy"
        if not logits_path.exists():
            raise FileNotFoundError(f"Missing Phase 4 logits: {logits_path}")

        logits = load_logits(logits_path)
        native_logits = resize_logits_to_shape(logits, native_shape)
        native_probs = 1.0 / (1.0 + np.exp(-native_logits))
        native_pred = (native_probs >= 0.5).astype(np.uint8)
        dice = dice_from_binary_masks(native_pred, label_arr)

        method_out_dir = output_root / method_name
        method_out_path = method_out_dir / f"{patient_id}_mask.nii.gz"
        write_native_prediction(native_pred, label_img, method_out_path)

        result["methods"][method_name] = {
            "phase4_logits_path": str(logits_path),
            "native_prediction_path": str(method_out_path),
            "logits_shape": list(logits.shape),
            "native_logits_shape": list(native_logits.shape),
            "dice": float(dice),
        }

    return result


def summarize(results: List[Dict[str, object]]) -> Dict[str, object]:
    method_names = sorted(results[0]["methods"].keys()) if results else []
    summary = {
        "num_patients": len(results),
        "methods": {},
    }

    for method_name in method_names:
        dices = [float(item["methods"][method_name]["dice"]) for item in results]
        summary["methods"][method_name] = {
            "mean_dice": float(np.mean(dices)) if dices else None,
            "median_dice": float(np.median(dices)) if dices else None,
            "min_dice": float(np.min(dices)) if dices else None,
            "max_dice": float(np.max(dices)) if dices else None,
            "num_patients": len(dices),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 reverse-alignment and evaluation.")
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="Process only the first N patients for a smoke test. Use 0 for all patients.",
    )
    args = parser.parse_args()

    method_dirs = {
        "simple_avg": PHASE4_DIR / "simple_avg",
        "lomix": PHASE4_DIR / "lomix",
        "dst": PHASE4_DIR / "dst",
    }
    for method_name, method_dir in method_dirs.items():
        if not method_dir.exists():
            raise FileNotFoundError(f"Missing Phase 4 output directory for {method_name}: {method_dir}")

    patient_ids = list_patient_ids(FEATURES_DIR)
    if args.max_patients > 0:
        patient_ids = patient_ids[: args.max_patients]

    output_root = OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for patient_id in patient_ids:
        results.append(process_patient(patient_id, method_dirs, output_root))

    summary = summarize(results)
    report = {
        "source": {
            "phase4_dir": str(PHASE4_DIR),
            "features_dir": str(FEATURES_DIR),
            "labels_dir": str(LABEL_DIR),
        },
        "num_patients": len(results),
        "methods": list(method_dirs.keys()),
        "patients": results,
        "summary": summary,
    }

    save_json(REPORT_DIR / "phase5_reverse_alignment.json", report)
    save_json(output_root / "phase5_native_summary.json", summary)

    print(f"Phase 5 reverse-alignment complete for {len(results)} patients.")
    for method_name, stats in summary["methods"].items():
        print(f"{method_name}: mean_dice={stats['mean_dice']:.4f} num_patients={stats['num_patients']}")


if __name__ == "__main__":
    main()