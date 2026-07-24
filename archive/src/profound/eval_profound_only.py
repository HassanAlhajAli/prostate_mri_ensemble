import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = WORKSPACE_DIR / "data" / "02_frozen_features" / "test"
LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTs"
OUTPUT_DIR = WORKSPACE_DIR / "data" / "05_profound_only_native_predictions"
REPORT_DIR = WORKSPACE_DIR / "reports"
DEFAULT_CKPT = WORKSPACE_DIR / "checkpoints" / "phase3" / "best.pth"


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProFoundDecoder3D(nn.Module):
    def __init__(self, in_chans: int = 768, target_size: int = 32) -> None:
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Conv3d(in_chans, 256, kernel_size=1)
        self.block1 = ConvBlock3D(256, 128)
        self.block2 = ConvBlock3D(128, 64)
        self.out_head = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(
            x,
            size=(self.target_size, self.target_size, self.target_size),
            mode="trilinear",
            align_corners=False,
        )
        x = self.block1(x)
        x = self.block2(x)
        return self.out_head(x)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def list_patient_ids(cache_dir: Path) -> List[str]:
    return sorted({p.name[: -len("_meta.json")] for p in cache_dir.glob("*_meta.json")})


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


def load_decoder(ckpt_path: Path, target_size: int, device: torch.device) -> ProFoundDecoder3D:
    decoder = ProFoundDecoder3D(in_chans=768, target_size=target_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt["model_state"]

    decoder_state = {}
    for key, value in model_state.items():
        if key.startswith("decoder."):
            decoder_state[key[len("decoder."):]] = value

    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing:
        print(f"Warning: missing decoder keys: {missing[:10]}")
    if unexpected:
        print(f"Warning: unexpected decoder keys: {unexpected[:10]}")

    decoder.eval()
    return decoder


def process_patient(
    patient_id: str,
    decoder: ProFoundDecoder3D,
    output_root: Path,
    threshold: float,
    device: torch.device,
) -> Dict[str, object]:
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

    profound_path = FEATURES_DIR / f"{patient_id}_profound.npy"
    if not profound_path.exists():
        raise FileNotFoundError(f"Missing ProFound feature file: {profound_path}")

    profound = np.load(profound_path).astype(np.float32)
    profound_t = torch.from_numpy(profound).unsqueeze(0).to(device)

    with torch.no_grad():
        profound_logits = decoder(profound_t)

    logits_np = profound_logits.squeeze(0).squeeze(0).cpu().numpy()
    native_logits = resize_logits_to_shape(logits_np, native_shape)
    native_probs = 1.0 / (1.0 + np.exp(-native_logits))
    native_pred = (native_probs >= threshold).astype(np.uint8)
    dice = dice_from_binary_masks(native_pred, label_arr)

    method_out_dir = output_root / "profound"
    method_out_path = method_out_dir / f"{patient_id}_mask.nii.gz"
    write_native_prediction(native_pred, label_img, method_out_path)

    return {
        "patient_id": patient_id,
        "native_shape": list(native_shape),
        "profound_feature_path": str(profound_path),
        "native_prediction_path": str(method_out_path),
        "profound_feature_shape": list(profound.shape),
        "canonical_logits_shape": list(logits_np.shape),
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
    parser = argparse.ArgumentParser(description="Standalone ProFound-only Dice evaluation on the fixed test split.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    patient_ids = list_patient_ids(FEATURES_DIR)
    if args.max_patients > 0:
        patient_ids = patient_ids[: args.max_patients]

    output_root = OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_decoder(ckpt_path, target_size=args.target_size, device=device)

    results: List[Dict[str, object]] = []
    for patient_id in patient_ids:
        results.append(process_patient(patient_id, decoder, output_root, args.threshold, device))

    summary = summarize(results)
    report = {
        "source": {
            "features_dir": str(FEATURES_DIR),
            "labels_dir": str(LABEL_DIR),
            "checkpoint": str(ckpt_path),
        },
        "num_patients": len(results),
        "model": "ProFound",
        "mode": "standalone",
        "target_size": int(args.target_size),
        "threshold": float(args.threshold),
        "patients": results,
        "summary": summary,
    }

    save_json(REPORT_DIR / "profound_only_eval.json", report)
    save_json(output_root / "profound_only_summary.json", summary)

    print(f"ProFound-only evaluation complete for {len(results)} patients.")
    print(f"Mean Dice: {summary['mean_dice']:.4f}")
    print(f"Outputs written under: {output_root}")


if __name__ == "__main__":
    main()
