import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROFOUND_SRC_DIR = WORKSPACE_DIR / "archive" / "ProFound"
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints" / "phase3"
FEATURE_DIR = WORKSPACE_DIR / "data" / "02_frozen_features"
TRAIN_FEATURE_DIR = FEATURE_DIR / "train"
TEST_FEATURE_DIR = FEATURE_DIR / "test"
DATASET_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS"
TRAIN_IMAGE_DIR = DATASET_DIR / "imagesTr"
TRAIN_LABEL_DIR = DATASET_DIR / "labelsTr"
TEST_IMAGE_DIR = DATASET_DIR / "imagesTs"
TEST_LABEL_DIR = DATASET_DIR / "labelsTs"
REPORT_DIR = WORKSPACE_DIR / "reports"
PREDICTION_DIR = REPORT_DIR / "phase3_predictions"

TARGET_SHAPE = (64, 160, 160)
DEFAULT_PROFOUND_CHECKPOINT = WORKSPACE_DIR / "archive" / "checkpoints" / "profound.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def load_profound_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(PROFOUND_SRC_DIR))
    try:
        from models.convnextv2 import convnextv2_tiny
    finally:
        sys.path.pop(0)

    model = convnextv2_tiny(in_chans=3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
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

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[profound] loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[profound] first missing keys: {missing[:10]}")
    if unexpected:
        print(f"[profound] first unexpected keys: {unexpected[:10]}")

    model.to(device).eval()
    return model


def read_volume(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    metadata = {
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }
    return array, metadata


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    if volume.size == 0:
        return volume
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    if not np.isfinite(std) or std < 1e-6:
        return volume - mean
    return (volume - mean) / std


def resize_array_to_shape(array: np.ndarray, target_shape: Tuple[int, int, int], mode: str = "trilinear") -> np.ndarray:
    if tuple(array.shape) == target_shape:
        return array
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def extract_profound_features_from_array(volume: np.ndarray, model: nn.Module, device: torch.device) -> np.ndarray:
    volume = normalize_volume(volume)
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
    profound_input = tensor.repeat(1, 3, 1, 1, 1)
    with torch.inference_mode():
        _, hidden_states = model(profound_input, ret_hids=True)
        features = hidden_states[-1].squeeze(0).cpu().numpy()
    return features.astype(np.float32)


def patient_id_from_image_name(path: Path) -> str:
    if path.name.endswith("_0000.nii.gz"):
        return path.name[: -len("_0000.nii.gz")]
    return path.name.replace(".nii.gz", "")


def extract_features(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}")
    model = load_profound_model(args.checkpoint, device)

    image_dirs = [
        (TRAIN_IMAGE_DIR, TRAIN_FEATURE_DIR, "train"),
        (TEST_IMAGE_DIR, TEST_FEATURE_DIR, "test"),
    ]

    for image_dir, output_dir, split_name in image_dirs:
        paths = sorted(image_dir.glob("*.nii.gz"))
        print(f"[extract] processing {split_name}: {len(paths)} scans")
        for path in tqdm(paths, desc=split_name):
            pid = patient_id_from_image_name(path)
            volume, _ = read_volume(path)
            features = extract_profound_features_from_array(volume, model, device)
            out_path = output_dir / f"{pid}_profound.npy"
            np.save(out_path, features)


class ProFoundDecoder3D(nn.Module):
    def __init__(self, in_channels: int = 768) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.out_head = nn.Conv3d(32, 1, kernel_size=1)
        nn.init.constant_(self.out_head.bias, -2.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.interpolate(x, size=(16, 40, 40), mode="trilinear", align_corners=False)
        x = self.block1(x)
        x = F.interpolate(x, size=(32, 80, 80), mode="trilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
        x = self.block3(x)
        return self.out_head(x)


class ProFoundDecoderDataset(Dataset):
    def __init__(self, feature_dir: Path, label_dir: Path, patient_ids: List[str]):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        feature_path = self.feature_dir / f"{pid}_profound.npy"
        label_path = self.label_dir / f"{pid}.nii.gz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature for {pid}: {feature_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {pid}: {label_path}")

        feature = np.load(feature_path).astype(np.float32)
        label_image = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label_image).astype(np.float32)
        label_array = (label_array > 0).astype(np.float32)

        if tuple(label_array.shape) != TARGET_SHAPE:
            label_array = resize_array_to_shape(label_array, TARGET_SHAPE, mode="nearest")

        feature_tensor = torch.from_numpy(feature).float()
        if feature_tensor.ndim == 4:
            # keep as [C, D, H, W]; DataLoader will add the batch dimension automatically
            pass
        elif feature_tensor.ndim != 5:
            raise ValueError(f"Unexpected feature shape for {pid}: {feature_tensor.shape}")

        label_tensor = torch.from_numpy(label_array).float().unsqueeze(0)
        return {"patient_id": pid, "feature": feature_tensor, "mask": label_tensor}


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, logits.dim()))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    loss = alpha * torch.pow(1.0 - p_t, gamma) * ce
    return loss.mean()


def train_model(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    patient_ids = sorted(set(feature_ids) & set(label_ids))
    if not patient_ids:
        raise RuntimeError("No matching feature/label pairs found for training")
    if args.overfit_n > 0:
        patient_ids = patient_ids[: min(args.overfit_n, len(patient_ids))]

    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, patient_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ProFoundDecoder3D(in_channels=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight = torch.tensor([args.pos_weight], device=device, dtype=torch.float32)

    best_path = CHECKPOINT_DIR / "best.pt"
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            feature = batch["feature"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if feature.dim() == 4:
                feature = feature.unsqueeze(0)
            if mask.dim() == 4:
                mask = mask.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast('cuda'):
                    logits = model(feature)
                    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)
                    dice = soft_dice_loss(logits, mask)
                    focal = focal_loss(logits, mask)
                    loss = args.bce_weight * bce + args.dice_weight * dice + args.focal_weight * focal
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(feature)
                bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)
                dice = soft_dice_loss(logits, mask)
                focal = focal_loss(logits, mask)
                loss = args.bce_weight * bce + args.dice_weight * dice + args.focal_weight * focal
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[train] epoch {epoch} loss={avg_loss:.4f}")

        checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
        }, checkpoint_path)

        if epoch == args.epochs:
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"[train] saved final checkpoint: {best_path}")

    print(f"[train] completed. checkpoint_dir={CHECKPOINT_DIR}")


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)
    if denom < 1e-8:
        dice = 0.0
    else:
        dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    precision = np.sum(pred * gt) / (np.sum(pred) + 1e-6)
    recall = np.sum(pred * gt) / (np.sum(gt) + 1e-6)
    return {"dice": float(dice), "precision": float(precision), "recall": float(recall)}


def evaluate_model(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    checkpoint_path = args.checkpoint if args.checkpoint else (CHECKPOINT_DIR / "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ProFoundDecoder3D(in_channels=768).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TEST_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TEST_LABEL_DIR.glob("*.nii.gz")})
    patient_ids = sorted(set(feature_ids) & set(label_ids))
    if not patient_ids:
        raise RuntimeError("No matching feature/label pairs found for evaluation")

    all_metrics: List[Dict[str, float]] = []
    for pid in tqdm(patient_ids, desc="evaluate"):
        feature_path = TEST_FEATURE_DIR / f"{pid}_profound.npy"
        image_path = TEST_IMAGE_DIR / f"{pid}.nii.gz"
        label_path = TEST_LABEL_DIR / f"{pid}.nii.gz"
        if not image_path.exists():
            image_path = TEST_IMAGE_DIR / f"{pid}_0000.nii.gz"
        if not image_path.exists():
            raise FileNotFoundError(f"No image found for {pid}")

        feature_array = np.load(feature_path).astype(np.float32)
        feature_tensor = torch.from_numpy(feature_array).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(feature_tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        pred_mask = (probs > args.threshold).astype(np.uint8)

        image = sitk.ReadImage(str(image_path))
        pred_img = sitk.GetImageFromArray(pred_mask.astype(np.float32))
        pred_img.SetSpacing(image.GetSpacing())
        pred_img.SetOrigin(image.GetOrigin())
        pred_img.SetDirection(image.GetDirection())

        pred_path = PREDICTION_DIR / f"{pid}_pred.nii.gz"
        sitk.WriteImage(pred_img, str(pred_path))

        label_array = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.float32)
        label_array = (label_array > 0).astype(np.uint8)
        metrics = compute_metrics(pred_mask.astype(np.float32), label_array.astype(np.float32))
        metrics["patient_id"] = pid
        all_metrics.append(metrics)

    mean_metrics = {
        "mean_dice": float(np.mean([m["dice"] for m in all_metrics])),
        "mean_precision": float(np.mean([m["precision"] for m in all_metrics])),
        "mean_recall": float(np.mean([m["recall"] for m in all_metrics])),
        "n_cases": len(all_metrics),
    }
    report_path = REPORT_DIR / "phase3_profound_baseline.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"metrics": mean_metrics, "per_case": all_metrics}, handle, indent=2)
    print(f"[eval] wrote report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ProFound decoder baseline")
    parser.add_argument("--mode", choices=["extract", "train", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PROFOUND_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overfit-n", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "extract":
        extract_features(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "evaluate":
        evaluate_model(args)


if __name__ == "__main__":
    main()