import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import torch

# Compatibility fix for older nnU-Net checkpoints on modern PyTorch
try:
    torch.serialization.add_safe_globals([
        np.dtype,
        np.bool_,
        np.int64,
        np.int32,
        np.float32,
        np.float64,
        np._core.multiarray.scalar,
        np.core.multiarray.scalar,
    ])
except Exception:
    pass

orig_torch_load = torch.load

def compat_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return orig_torch_load(*args, **kwargs)

torch.load = compat_load

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS"
PREPROCESSED_DIR = ROOT / "data" / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset500_PROMIS"
RESULTS_DIR = ROOT / "data" / "nnUNet_data" / "nnUNet_results"
MODEL_DIR = RESULTS_DIR / "Dataset500_PROMIS" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
PREDICTIONS_DIR = RESULTS_DIR / "predictions_test_final"
REPORT_PATH = ROOT / "reports" / "phase2_nnunet_baseline.json"


def ensure_model_files() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fold_dir = MODEL_DIR / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for src_name in ["checkpoint_final.pth", "checkpoint_best.pth"]:
        src_path = ROOT / src_name
        if src_path.exists():
            dst_path = fold_dir / src_name
            if not dst_path.exists() or dst_path.stat().st_size != src_path.stat().st_size:
                shutil.copy2(src_path, dst_path)

    for src_name in ["plans.json", "dataset.json"]:
        src_path = PREPROCESSED_DIR / src_name
        if src_path.exists():
            dst_path = MODEL_DIR / src_name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)


def get_image_and_label_files() -> List[Tuple[Path, Path]]:
    image_dir = RAW_DATA_DIR / "imagesTs"
    label_dir = RAW_DATA_DIR / "labelsTs"
    image_files = sorted(image_dir.glob("*_0000.nii.gz"))
    pairs: List[Tuple[Path, Path]] = []
    for image_path in image_files:
        case_id = image_path.name.replace("_0000.nii.gz", "")
        label_path = label_dir / f"{case_id}.nii.gz"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"Missing label for {case_id}")
    return pairs


def find_prediction_file(output_dir: Path, case_id: str) -> Path:
    candidates = [
        output_dir / f"{case_id}.nii.gz",
        output_dir / f"{case_id}_0000.nii.gz",
        output_dir / f"{case_id}_seg.nii.gz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    for cand in sorted(output_dir.glob("*.nii.gz")):
        if case_id in cand.name:
            return cand
    return output_dir / f"{case_id}.nii.gz"


def load_array(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = binarize(pred)
    target_b = binarize(target)
    inter = np.sum(pred_b & target_b)
    denom = np.sum(pred_b) + np.sum(target_b)
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2.0 * inter / denom)


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = binarize(pred)
    target_b = binarize(target)
    inter = np.sum(pred_b & target_b)
    union = np.sum(pred_b | target_b)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_b = binarize(pred)
    target_b = binarize(target)
    tp = int(np.sum((pred_b == 1) & (target_b == 1)))
    fp = int(np.sum((pred_b == 1) & (target_b == 0)))
    fn = int(np.sum((pred_b == 0) & (target_b == 1)))
    tn = int(np.sum((pred_b == 0) & (target_b == 0)))
    sens = tp / (tp + fn) if (tp + fn) else 1.0
    spec = tn / (tn + fp) if (tn + fp) else 1.0
    return {
        "dice": compute_dice(pred, target),
        "iou": compute_iou(pred, target),
        "sensitivity": sens,
        "specificity": spec,
    }


def run_inference(image_files: List[Path]) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        [[str(p)] for p in image_files],
        str(PREDICTIONS_DIR),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )


def evaluate_predictions() -> dict:
    pairs = get_image_and_label_files()
    per_case: List[dict] = []
    dices: List[float] = []
    for image_path, label_path in pairs:
        case_id = image_path.name.replace("_0000.nii.gz", "")
        pred_path = find_prediction_file(PREDICTIONS_DIR, case_id)
        if not pred_path.exists():
            print(f"Prediction missing for {case_id}")
            continue
        pred_arr = load_array(pred_path)
        target_arr = load_array(label_path)
        metrics = compute_metrics(pred_arr, target_arr)
        per_case.append({
            "case_id": case_id,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
        })
        dices.append(metrics["dice"])

    return {
        "checkpoint": "checkpoint_final.pth",
        "num_cases": len(per_case),
        "mean_dice": float(np.mean(dices)) if dices else None,
        "median_dice": float(np.median(dices)) if dices else None,
        "min_dice": float(np.min(dices)) if dices else None,
        "max_dice": float(np.max(dices)) if dices else None,
        "per_case": per_case,
    }


def main() -> None:
    ensure_model_files()
    image_and_label_pairs = get_image_and_label_files()
    image_files = [img for img, _ in image_and_label_pairs]
    if not image_files:
        raise RuntimeError("No test images found for evaluation")
    run_inference(image_files)
    report = evaluate_predictions()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
