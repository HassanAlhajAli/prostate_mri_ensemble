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
LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTs"
OUTPUT_DIR = WORKSPACE_DIR / "data" / "05_medsam3d_only_native_predictions"
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


def write_native_prediction(pred_mask: np.ndarray, reference_image: sitk.Image, output_path: Path) -> None:
	pred_img = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
	pred_img.CopyInformation(reference_image)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	sitk.WriteImage(pred_img, str(output_path))


def process_patient(patient_id: str, output_root: Path, threshold: float) -> Dict[str, object]:
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

	logits_path = FEATURES_DIR / f"{patient_id}_sam_logits.npy"
	if not logits_path.exists():
		raise FileNotFoundError(f"Missing MedSAM3D logits: {logits_path}")

	logits = load_logits(logits_path)
	native_logits = resize_logits_to_shape(logits, native_shape)
	native_probs = 1.0 / (1.0 + np.exp(-native_logits))
	native_pred = (native_probs >= threshold).astype(np.uint8)
	dice = dice_from_binary_masks(native_pred, label_arr)

	method_out_dir = output_root / "medsam3d"
	method_out_path = method_out_dir / f"{patient_id}_mask.nii.gz"
	write_native_prediction(native_pred, label_img, method_out_path)

	return {
		"patient_id": patient_id,
		"native_shape": list(native_shape),
		"sam_logits_path": str(logits_path),
		"native_prediction_path": str(method_out_path),
		"logits_shape": list(logits.shape),
		"native_logits_shape": list(native_logits.shape),
		"threshold": float(threshold),
		"dice": float(dice),
	}


def summarize(results: List[Dict[str, object]]) -> Dict[str, object]:
	dices = [float(item["dice"]) for item in results]
	return {
		"num_patients": len(results),
		"mean_dice": float(np.mean(dices)) if dices else None,
		"median_dice": float(np.median(dices)) if dices else None,
		"min_dice": float(np.min(dices)) if dices else None,
		"max_dice": float(np.max(dices)) if dices else None,
		"std_dice": float(np.std(dices)) if dices else None,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Standalone MedSAM3D Dice evaluation on the fixed test split.")
	parser.add_argument(
		"--max-patients",
		type=int,
		default=0,
		help="Process only the first N patients for a smoke test. Use 0 for all patients.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.5,
		help="Binary threshold applied after sigmoid.",
	)
	args = parser.parse_args()

	patient_ids = list_patient_ids(FEATURES_DIR)
	if args.max_patients > 0:
		patient_ids = patient_ids[: args.max_patients]

	output_root = OUTPUT_DIR
	output_root.mkdir(parents=True, exist_ok=True)

	results: List[Dict[str, object]] = []
	for patient_id in patient_ids:
		results.append(process_patient(patient_id, output_root, args.threshold))

	summary = summarize(results)
	report = {
		"source": {
			"features_dir": str(FEATURES_DIR),
			"labels_dir": str(LABEL_DIR),
		},
		"num_patients": len(results),
		"model": "MedSAM3D",
		"threshold": float(args.threshold),
		"patients": results,
		"summary": summary,
	}

	save_json(REPORT_DIR / "medsam3d_only_eval.json", report)
	save_json(output_root / "medsam3d_only_summary.json", summary)

	print(f"MedSAM3D-only evaluation complete for {len(results)} patients.")
	print(f"Mean Dice: {summary['mean_dice']:.4f}")
	print(f"Outputs written under: {output_root}")


if __name__ == "__main__":
	main()