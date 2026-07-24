import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = WORKSPACE_DIR / "data" / "02_frozen_features" / "test"
LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTs"
PRED_DIR = WORKSPACE_DIR / "data" / "05_medsam3d_only_native_predictions" / "medsam3d"
OUTPUT_DIR = WORKSPACE_DIR / "reports" / "medsam3d_case_viz"


def list_patient_ids(cache_dir: Path) -> List[str]:
	return sorted({p.name[: -len("_meta.json")] for p in cache_dir.glob("*_meta.json")})


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def load_binary_mask(path: Path) -> np.ndarray:
	img = sitk.ReadImage(str(path))
	arr = sitk.GetArrayFromImage(img).astype(np.float32)
	return (arr > 0).astype(np.uint8)


def dice_from_binary_masks(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
	pred = pred.astype(np.float32)
	target = target.astype(np.float32)
	intersection = float((pred * target).sum())
	denom = float(pred.sum() + target.sum())
	return float((2.0 * intersection + eps) / (denom + eps))


def pick_max_slice(mask: np.ndarray, axis: int) -> int:
	if mask.sum() == 0:
		return mask.shape[axis] // 2
	projection = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis))
	return int(np.argmax(projection))


def make_overlay(gt_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
	overlay = np.zeros(gt_slice.shape + (3,), dtype=np.float32)
	overlay[..., 0] = gt_slice.astype(np.float32)
	overlay[..., 1] = pred_slice.astype(np.float32)
	return overlay


def save_slice_figure(gt: np.ndarray, pred: np.ndarray, out_path: Path, patient_id: str, dice: float) -> None:
	axis_specs = [
		("axial", 0),
		("coronal", 1),
		("sagittal", 2),
	]

	fig, axes = plt.subplots(3, 3, figsize=(12, 12))
	fig.suptitle(f"MedSAM3D case view: {patient_id} | Dice={dice:.4f}", fontsize=14)

	for row, (axis_name, axis) in enumerate(axis_specs):
		idx = pick_max_slice(gt, axis)
		if axis == 0:
			gt_slice = gt[idx, :, :]
			pred_slice = pred[idx, :, :]
		elif axis == 1:
			gt_slice = gt[:, idx, :]
			pred_slice = pred[:, idx, :]
		else:
			gt_slice = gt[:, :, idx]
			pred_slice = pred[:, :, idx]

		overlay = make_overlay(gt_slice, pred_slice)

		axes[row, 0].imshow(gt_slice, cmap="gray")
		axes[row, 0].set_title(f"{axis_name} GT")
		axes[row, 1].imshow(pred_slice, cmap="gray")
		axes[row, 1].set_title(f"{axis_name} Pred")
		axes[row, 2].imshow(overlap := overlay)
		axes[row, 2].set_title(f"{axis_name} Overlay")

		for col in range(3):
			axes[row, col].axis("off")

	fig.tight_layout(rect=[0, 0.03, 1, 0.96])
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize one MedSAM3D case as binary masks and slice panels.")
	parser.add_argument(
		"--patient-id",
		type=str,
		default="",
		help="Patient ID to visualize. Defaults to the first test patient.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(OUTPUT_DIR),
		help="Directory where visualizations will be written.",
	)
	args = parser.parse_args()

	patient_ids = list_patient_ids(FEATURES_DIR)
	if not patient_ids:
		raise RuntimeError(f"No patient metadata found in {FEATURES_DIR}")

	patient_id = args.patient_id or patient_ids[0]
	if patient_id not in patient_ids:
		raise FileNotFoundError(f"Patient '{patient_id}' not found in the test cache.")

	meta_path = FEATURES_DIR / f"{patient_id}_meta.json"
	meta = load_json(meta_path)
	native_shape = meta.get("native_image", {}).get("shape_dhw")
	if not native_shape:
		raise ValueError(f"Missing native shape in metadata for {patient_id}")

	label_path = LABEL_DIR / f"{patient_id}.nii.gz"
	pred_path = PRED_DIR / f"{patient_id}_mask.nii.gz"
	if not label_path.exists():
		raise FileNotFoundError(f"Missing label file: {label_path}")
	if not pred_path.exists():
		raise FileNotFoundError(f"Missing prediction file: {pred_path}")

	gt = load_binary_mask(label_path)
	pred = load_binary_mask(pred_path)
	if list(gt.shape) != list(native_shape):
		raise ValueError(f"GT shape mismatch for {patient_id}: {list(gt.shape)} vs {native_shape}")
	if list(pred.shape) != list(native_shape):
		raise ValueError(f"Prediction shape mismatch for {patient_id}: {list(pred.shape)} vs {native_shape}")

	dice = dice_from_binary_masks(pred, gt)
	out_dir = Path(args.output_dir) / patient_id
	out_dir.mkdir(parents=True, exist_ok=True)

	np.save(out_dir / "gt_binary.npy", gt.astype(np.uint8))
	np.save(out_dir / "pred_binary.npy", pred.astype(np.uint8))

	fig_path = out_dir / "medsam3d_case_slices.png"
	save_slice_figure(gt, pred, fig_path, patient_id, dice)

	report = {
		"patient_id": patient_id,
		"native_shape": list(native_shape),
		"label_path": str(label_path),
		"prediction_path": str(pred_path),
		"dice": float(dice),
		"outputs": {
			"gt_binary_npy": str(out_dir / "gt_binary.npy"),
			"pred_binary_npy": str(out_dir / "pred_binary.npy"),
			"figure": str(fig_path),
		},
	}

	with open(out_dir / "case_report.json", "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	print(f"Visualized patient: {patient_id}")
	print(f"Dice: {dice:.4f}")
	print(f"Figure: {fig_path}")


if __name__ == "__main__":
	main()