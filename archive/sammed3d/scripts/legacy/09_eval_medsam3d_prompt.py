import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
SAM_MED3D_DIR = WORKSPACE_DIR / "SAM-Med3D"
TEST_IMAGE_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "imagesTs"
TEST_LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTs"
OUTPUT_DIR = WORKSPACE_DIR / "data" / "05_medsam3d_prompt_native_predictions"
REPORT_DIR = WORKSPACE_DIR / "reports"
CHECKPOINT_PATH = WORKSPACE_DIR / "checkpoints" / "sam_med3d_turbo.pth"


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def list_patient_ids(image_dir: Path) -> List[str]:
	patient_ids = []
	for path in sorted(image_dir.glob("*.nii.gz")):
		if path.name.endswith("_0000.nii.gz"):
			patient_ids.append(path.name[: -len("_0000.nii.gz")])
		else:
			patient_ids.append(path.name[: -len(".nii.gz")])
	return patient_ids


def load_binary_label(path: Path) -> Tuple[np.ndarray, sitk.Image]:
	label_img = sitk.ReadImage(str(path))
	label_arr = sitk.GetArrayFromImage(label_img).astype(np.float32)
	label_arr = (label_arr > 0).astype(np.uint8)
	return label_arr, label_img


def dice_from_binary_masks(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
	pred = pred.astype(np.float32)
	target = target.astype(np.float32)
	intersection = float((pred * target).sum())
	denom = float(pred.sum() + target.sum())
	return float((2.0 * intersection + eps) / (denom + eps))


def load_sam_checkpoint(checkpoint_path: Path) -> dict:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	if isinstance(checkpoint, dict):
		if "model_state_dict" in checkpoint:
			state_dict = checkpoint["model_state_dict"]
		elif "model" in checkpoint:
			state_dict = checkpoint["model"]
		else:
			state_dict = checkpoint
	else:
		state_dict = checkpoint

	if isinstance(state_dict, dict):
		cleaned_state_dict = {}
		for key, value in state_dict.items():
			if key.startswith("module."):
				cleaned_state_dict[key[len("module."):]] = value
			else:
				cleaned_state_dict[key] = value
		state_dict = cleaned_state_dict
	return state_dict


def build_model():
	sys.path.insert(0, str(SAM_MED3D_DIR))
	from segment_anything.build_sam3D import build_sam3D_vit_b_ori

	model = build_sam3D_vit_b_ori(checkpoint=None)
	state_dict = load_sam_checkpoint(CHECKPOINT_PATH)
	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	print(f"Loaded SAM-Med3D checkpoint with missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
	if missing_keys:
		print(f"First missing keys: {missing_keys[:10]}")
	if unexpected_keys:
		print(f"First unexpected keys: {unexpected_keys[:10]}")
	return model


def load_helpers():
	sys.path.insert(0, str(SAM_MED3D_DIR))
	from utils.infer_utils import data_postprocess, random_sample_next_click
	return data_postprocess, random_sample_next_click


def preprocess_subject(
	img_path: Path,
	gt_path: Path,
	target_spacing: Tuple[float, float, float],
	crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
	subject = tio.Subject(
		image=tio.ScalarImage(str(img_path)),
		label=tio.LabelMap(str(gt_path)),
	)

	label_data_for_cat = subject.label.data.clone()
	new_label_data = torch.zeros_like(label_data_for_cat)
	new_label_data[label_data_for_cat > 0] = 1
	subject.label.set_data(new_label_data)

	meta_info = {
		"original_subject_affine": subject.image.affine.copy(),
		"original_subject_spatial_shape": subject.image.spatial_shape,
	}

	resampler = tio.Resample(target_spacing)
	subject_resampled = resampler(subject)
	transform_canonical = tio.ToCanonical()
	subject_canonical = transform_canonical(subject_resampled)

	crop_transform = tio.CropOrPad(mask_name="label", target_shape=(crop_size, crop_size, crop_size))
	subject_cropped = crop_transform(subject_canonical)

	meta_info["canonical_subject_shape"] = subject_canonical.spatial_shape
	meta_info["canonical_subject_affine"] = subject_canonical.image.affine.copy()
	meta_info["roi_subject_affine"] = subject_cropped.image.affine.copy()

	img3d_roi = subject_cropped.image.data.clone().detach()
	img3d_roi = (img3d_roi - img3d_roi.mean()) / img3d_roi.std().clamp_min(1e-6)
	img3d_roi = img3d_roi.unsqueeze(dim=1) if img3d_roi.ndim == 4 else img3d_roi
	if img3d_roi.ndim == 4:
		img3d_roi = img3d_roi.unsqueeze(dim=0)
	if img3d_roi.ndim == 5 and img3d_roi.shape[0] != 1:
		img3d_roi = img3d_roi[:, 0:1, ...]

	gt3d_roi = subject_cropped.label.data.clone().detach()
	if gt3d_roi.ndim == 4:
		gt3d_roi = gt3d_roi.unsqueeze(dim=0)
	if gt3d_roi.ndim == 5 and gt3d_roi.shape[0] != 1:
		gt3d_roi = gt3d_roi[:, 0:1, ...]

	return img3d_roi, gt3d_roi, meta_info


def sam_model_infer_prompt(model, roi_image: torch.Tensor, roi_gt: torch.Tensor, random_sample_next_click, num_clicks: int = 1, prev_low_res_mask=None):
	model.eval()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	with torch.no_grad():
		input_tensor = roi_image.to(device)
		image_embeddings = model.image_encoder(input_tensor)

		points_coords = torch.zeros(1, 0, 3, device=device)
		points_labels = torch.zeros(1, 0, device=device, dtype=torch.long)
		current_prev_mask_for_click_generation = torch.zeros_like(roi_image, device=device)[:, 0, ...]

		if prev_low_res_mask is None:
			prev_low_res_mask = torch.zeros(
				1,
				1,
				roi_image.shape[2] // 4,
				roi_image.shape[3] // 4,
				roi_image.shape[4] // 4,
				device=device,
				dtype=torch.float,
			)

		low_res_masks = prev_low_res_mask
		for _ in range(num_clicks):
			new_points_co, new_points_la = random_sample_next_click(
				current_prev_mask_for_click_generation.squeeze(0).cpu(),
				roi_gt[0, 0].cpu(),
			)
			new_points_co, new_points_la = new_points_co.to(device), new_points_la.to(device)

			points_coords = torch.cat([points_coords, new_points_co], dim=1)
			points_labels = torch.cat([points_labels, new_points_la], dim=1)

			sparse_embeddings, dense_embeddings = model.prompt_encoder(
				points=[points_coords, points_labels],
				boxes=None,
				masks=prev_low_res_mask,
			)

			low_res_masks, _ = model.mask_decoder(
				image_embeddings=image_embeddings,
				image_pe=model.prompt_encoder.get_dense_pe(),
				sparse_prompt_embeddings=sparse_embeddings,
				dense_prompt_embeddings=dense_embeddings,
				multimask_output=False,
			)
			prev_low_res_mask = low_res_masks.detach()
			current_prev_mask_for_click_generation = F.interpolate(
				low_res_masks,
				size=roi_image.shape[-3:],
				mode="trilinear",
				align_corners=False,
			)
			current_prev_mask_for_click_generation = torch.sigmoid(current_prev_mask_for_click_generation) > 0.5

		final_masks_hr = F.interpolate(
			low_res_masks,
			size=roi_image.shape[-3:],
				mode="trilinear",
			align_corners=False,
		)

	medsam_seg_prob = torch.sigmoid(final_masks_hr)
	medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
	medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)
	return medsam_seg_mask, low_res_masks.detach()


def process_patient(
	patient_id: str,
	model,
	data_postprocess,
	random_sample_next_click,
	output_root: Path,
	target_spacing: Tuple[float, float, float],
	crop_size: int,
	num_clicks: int,
) -> Dict[str, object]:
	img_path = TEST_IMAGE_DIR / f"{patient_id}_0000.nii.gz"
	gt_path = TEST_LABEL_DIR / f"{patient_id}.nii.gz"
	if not img_path.exists():
		raise FileNotFoundError(f"Missing test image: {img_path}")
	if not gt_path.exists():
		raise FileNotFoundError(f"Missing test label: {gt_path}")

	label_arr, label_img = load_binary_label(gt_path)
	label_sum = int(label_arr.sum())
	if label_sum == 0:
		empty_pred = np.zeros_like(label_arr, dtype=np.uint8)
		dice = dice_from_binary_masks(empty_pred, label_arr)
		method_out_dir = output_root / "medsam3d"
		method_out_dir.mkdir(parents=True, exist_ok=True)
		method_out_path = method_out_dir / f"{patient_id}_mask.nii.gz"
		out_img = sitk.GetImageFromArray(empty_pred.astype(np.uint8))
		out_img.CopyInformation(label_img)
		sitk.WriteImage(out_img, str(method_out_path))
		return {
			"patient_id": patient_id,
			"image_path": str(img_path),
			"label_path": str(gt_path),
			"native_prediction_path": str(method_out_path),
			"native_shape": list(label_arr.shape),
			"roi_shape": None,
			"roi_label_shape": None,
			"low_res_mask_shape": None,
			"target_spacing": list(target_spacing),
			"crop_size": int(crop_size),
			"num_clicks": int(num_clicks),
			"label_sum": label_sum,
			"empty_label_case": True,
			"dice": float(dice),
		}

	roi_image, roi_label, meta_info = preprocess_subject(img_path, gt_path, target_spacing, crop_size)

	roi_pred_numpy, low_res_mask = sam_model_infer_prompt(
		model,
		roi_image,
		roi_label,
		random_sample_next_click,
		num_clicks=num_clicks,
		prev_low_res_mask=None,
	)

	pred_original_grid = data_postprocess(roi_pred_numpy, meta_info)
	pred_original_grid = (pred_original_grid > 0).astype(np.uint8)
	if list(pred_original_grid.shape) != list(label_arr.shape):
		raise ValueError(
			f"Prediction shape mismatch for {patient_id}: pred={list(pred_original_grid.shape)} label={list(label_arr.shape)}"
		)

	dice = dice_from_binary_masks(pred_original_grid, label_arr)

	method_out_dir = output_root / "medsam3d"
	method_out_dir.mkdir(parents=True, exist_ok=True)
	method_out_path = method_out_dir / f"{patient_id}_mask.nii.gz"
	out_img = sitk.GetImageFromArray(pred_original_grid.astype(np.uint8))
	out_img.CopyInformation(label_img)
	sitk.WriteImage(out_img, str(method_out_path))

	return {
		"patient_id": patient_id,
		"image_path": str(img_path),
		"label_path": str(gt_path),
		"native_prediction_path": str(method_out_path),
		"native_shape": list(label_arr.shape),
		"roi_shape": list(roi_image.shape),
		"roi_label_shape": list(roi_label.shape),
		"low_res_mask_shape": list(low_res_mask.shape) if low_res_mask is not None else None,
		"target_spacing": list(target_spacing),
		"crop_size": int(crop_size),
		"num_clicks": int(num_clicks),
		"label_sum": label_sum,
		"empty_label_case": False,
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
	parser = argparse.ArgumentParser(description="Prompt-driven MedSAM3D evaluation on the fixed test split.")
	parser.add_argument("--max-patients", type=int, default=0)
	parser.add_argument("--target-spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
	parser.add_argument("--crop-size", type=int, default=128)
	parser.add_argument("--num-clicks", type=int, default=1)
	args = parser.parse_args()

	if not CHECKPOINT_PATH.exists():
		raise FileNotFoundError(f"Missing MedSAM3D checkpoint: {CHECKPOINT_PATH}")

	patient_ids = list_patient_ids(TEST_IMAGE_DIR)
	if args.max_patients > 0:
		patient_ids = patient_ids[: args.max_patients]

	model = build_model()
	model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	model.eval()
	data_postprocess, random_sample_next_click = load_helpers()

	output_root = OUTPUT_DIR
	output_root.mkdir(parents=True, exist_ok=True)

	results: List[Dict[str, object]] = []
	for patient_id in patient_ids:
		results.append(
			process_patient(
				patient_id,
				model,
				data_postprocess,
				random_sample_next_click,
				output_root,
				tuple(args.target_spacing),
				args.crop_size,
				args.num_clicks,
			)
		)

	summary = summarize(results)
	report = {
		"source": {
			"image_dir": str(TEST_IMAGE_DIR),
			"label_dir": str(TEST_LABEL_DIR),
			"checkpoint": str(CHECKPOINT_PATH),
		},
		"model": "MedSAM3D",
		"num_patients": len(results),
		"target_spacing": list(args.target_spacing),
		"crop_size": int(args.crop_size),
		"num_clicks": int(args.num_clicks),
		"patients": results,
		"summary": summary,
	}

	save_json(REPORT_DIR / "medsam3d_prompt_eval.json", report)
	save_json(output_root / "medsam3d_prompt_summary.json", summary)

	print(f"MedSAM3D prompt evaluation complete for {len(results)} patients.")
	print(f"Mean Dice: {summary['mean_dice']:.4f}")
	print(f"Outputs written under: {output_root}")


if __name__ == "__main__":
	main()