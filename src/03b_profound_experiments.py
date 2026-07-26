import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==============================================================================
# 1. PATHS & DIRECTORIES CONFIGURATION
# ==============================================================================
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
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


# ==============================================================================
# 2. DATASET (WITH IN-MEMORY CACHING) & UTILITIES
# ==============================================================================
def resize_array_to_shape(array: np.ndarray, target_shape: Tuple[int, int, int], mode: str = "nearest") -> np.ndarray:
    if tuple(array.shape) == target_shape:
        return array
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False if mode == "trilinear" else None)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


class ProFoundDecoderDataset(Dataset):
    def __init__(self, feature_dir: Path, label_dir: Path, patient_ids: List[str], use_cache: bool = True):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.patient_ids = patient_ids
        self.use_cache = use_cache
        self.cache = {}
        
        # Load everything into RAM once before training starts to bypass WSL I/O bottlenecks
        if self.use_cache:
            for pid in tqdm(self.patient_ids, desc="Caching dataset to RAM", leave=False):
                self.cache[pid] = self._load_from_disk(pid)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_from_disk(self, pid: str) -> Dict[str, torch.Tensor]:
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
        label_tensor = torch.from_numpy(label_array).float().unsqueeze(0)
        return {"patient_id": pid, "feature": feature_tensor, "mask": label_tensor}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        if self.use_cache:
            return self.cache[pid]  # Instant RAM access (No disk I/O!)
        return self._load_from_disk(pid)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)
    dice = 0.0 if denom < 1e-8 else (2.0 * intersection + 1e-6) / (denom + 1e-6)
    precision = np.sum(pred * gt) / (np.sum(pred) + 1e-6)
    recall = np.sum(pred * gt) / (np.sum(gt) + 1e-6)
    return {"dice": float(dice), "precision": float(precision), "recall": float(recall)}


# ==============================================================================
# 3. ADVANCED DECODER & LOSS FUNCTIONS
# ==============================================================================
class AdvancedProFoundDecoder3D(nn.Module):
    def __init__(self, in_channels: int = 768, base_channels: int = 64, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, base_channels, kernel_size=1)
        self.conv1 = self._make_block(base_channels, base_channels // 2, dropout_p)
        self.conv2 = self._make_block(base_channels // 2, base_channels // 4, dropout_p)
        self.conv3 = self._make_block(base_channels // 4, base_channels // 8, dropout_p)
        self.conv4 = self._make_block(base_channels // 8, base_channels // 16, dropout_p)
        self.out_head = nn.Conv3d(base_channels // 16, 1, kernel_size=1)
        nn.init.constant_(self.out_head.bias, -2.5)

    def _make_block(self, in_c: int, out_c: int, drop_p: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=drop_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.conv1(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv2(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv3(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv4(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        return self.out_head(x)


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
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    loss = alpha * torch.pow(1.0 - p_t, gamma) * ce
    return loss.mean()


# ==============================================================================
# 4. SINGLE TRIAL / EXPERIMENT RUNNER (WITH PROGRESS BARS)
# ==============================================================================
def run_single_experiment(trial_id: int, config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> float:
    set_seed(config["seed"])
    
    model = AdvancedProFoundDecoder3D(
        in_channels=768, 
        base_channels=config["decoder_channels"], 
        dropout_p=config["dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight = torch.tensor([config["pos_weight"]], device=device, dtype=torch.float32)

    best_val_dice = -1.0
    trial_checkpoint_path = CHECKPOINT_DIR / f"trial_{trial_id:02d}_best.pt"

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        
        # Added tqdm progress bar for batch training
        batch_iterator = tqdm(train_loader, desc=f"Trial {trial_id:02d} | Ep {epoch:02d}/{config['epochs']}", leave=False)
        for batch in batch_iterator:
            feature = batch["feature"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if feature.dim() == 4: feature = feature.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast('cuda'):
                    logits = model(feature)
                    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)
                    dice = soft_dice_loss(logits, mask)
                    focal = focal_loss(logits, mask)
                    loss = config["bce_weight"] * bce + config["dice_weight"] * dice + config["focal_weight"] * focal
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(feature)
                bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)
                dice = soft_dice_loss(logits, mask)
                focal = focal_loss(logits, mask)
                loss = config["bce_weight"] * bce + config["dice_weight"] * dice + config["focal_weight"] * focal
                loss.backward()
                optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            batch_iterator.set_postfix(loss=f"{loss_val:.4f}")

        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_dices = []
        with torch.inference_mode():
            for batch in val_loader:
                feature = batch["feature"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                if feature.dim() == 4: feature = feature.unsqueeze(0)
                if mask.dim() == 4: mask = mask.unsqueeze(0)
                
                logits = model(feature)
                preds = (torch.sigmoid(logits) > config["threshold"]).float()
                intersection = (preds * mask).sum()
                denom = preds.sum() + mask.sum()
                dice_score = (2.0 * intersection) / (denom + 1e-6)
                val_dices.append(dice_score.item())

        avg_val_dice = sum(val_dices) / max(1, len(val_dices))
        is_best = avg_val_dice > best_val_dice
        if is_best:
            best_val_dice = avg_val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_dice": best_val_dice
            }, trial_checkpoint_path)

        print(f"  [Trial {trial_id:02d}] Epoch {epoch:02d}/{config['epochs']} completed | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}{' (New Best!)' if is_best else ''}")

    print(f"[Trial {trial_id:02d}] Finished | Best Val Dice: {best_val_dice:.4f}\n")
    return best_val_dice


def sweep_experiments(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] Starting automated 10-trial hyperparameter sweep on device={device}")

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    patient_ids = sorted(set(feature_ids) & set(label_ids))

    random.seed(args.seed)
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)
    train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]

    # The dataset now explicitly uses the cache!
    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids, use_cache=True)
    val_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids, use_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    search_space = {
        "lr": [1e-4, 3e-4, 5e-4],
        "decoder_channels": [32, 64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "pos_weight": [2.0, 3.0, 4.0],
        "dice_weight": [1.0, 1.5, 2.0],
        "bce_weight": [0.5],
        "focal_weight": [0.5],
        "epochs": [30],
        "threshold": [0.3, 0.4, 0.5],
        "seed": [args.seed]
    }

    trials = []
    for i in range(1, 11):
        config = {k: random.choice(v) for k, v in search_space.items()}
        trials.append((i, config))

    results = []
    best_overall_dice = -1.0
    best_config = None

    for trial_id, config in trials:
        print(f"\n--- Running Trial {trial_id}/10 with config: {config} ---")
        val_dice = run_single_experiment(trial_id, config, train_loader, val_loader, device)
        results.append({"trial_id": trial_id, "val_dice": val_dice, "config": config})
        
        if val_dice > best_overall_dice:
            best_overall_dice = val_dice
            best_config = config
            src_path = CHECKPOINT_DIR / f"trial_{trial_id:02d}_best.pt"
            dst_path = CHECKPOINT_DIR / "best_advanced.pt"
            if src_path.exists():
                torch.save(torch.load(src_path, weights_only=False), dst_path)

    summary_path = REPORT_DIR / "phase3_hyperparameter_sweep.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_dice": best_overall_dice, "best_config": best_config, "trials": results}, f, indent=2)

    print("\n==============================================")
    print(f" SWEEP COMPLETE! Best Val Dice: {best_overall_dice:.4f}")
    print(f" Best Config: {best_config}")
    print("==============================================\n")


# ==============================================================================
# 5. STANDARD TRAINING & EVALUATION LOOPS
# ==============================================================================
def train_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[experiment] device={device} | base_channels={args.decoder_channels} | dropout={args.dropout}")

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    patient_ids = sorted(set(feature_ids) & set(label_ids))
    
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    if args.overfit_n > 0:
        train_ids = train_ids[: min(args.overfit_n, len(train_ids))]
        val_ids = train_ids

    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids, use_cache=True)
    val_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids, use_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = AdvancedProFoundDecoder3D(in_channels=768, base_channels=args.decoder_channels, dropout_p=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    pos_weight = torch.tensor([args.pos_weight], device=device, dtype=torch.float32)
    best_path = CHECKPOINT_DIR / "best_advanced.pt"
    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False):
            feature = batch["feature"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if feature.dim() == 4: feature = feature.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)

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

        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_dices = []
        with torch.inference_mode():
            for batch in val_loader:
                feature = batch["feature"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                if feature.dim() == 4: feature = feature.unsqueeze(0)
                if mask.dim() == 4: mask = mask.unsqueeze(0)
                
                logits = model(feature)
                preds = (torch.sigmoid(logits) > args.threshold).float()
                
                intersection = (preds * mask).sum()
                denom = preds.sum() + mask.sum()
                dice_score = (2.0 * intersection) / (denom + 1e-6)
                val_dices.append(dice_score.item())

        avg_val_dice = sum(val_dices) / max(1, len(val_dices))
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": best_val_dice,
            }, best_path)
            print(f"  -> New best model saved! (Val Dice: {best_val_dice:.4f})")


def evaluate_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    checkpoint_path = args.checkpoint if args.checkpoint else (CHECKPOINT_DIR / "best_advanced.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    base_channels = state.get("config", {}).get("decoder_channels", args.decoder_channels)
    dropout = state.get("config", {}).get("dropout", args.dropout)

    model = AdvancedProFoundDecoder3D(in_channels=768, base_channels=base_channels, dropout_p=dropout).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TEST_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TEST_LABEL_DIR.glob("*.nii.gz")})
    patient_ids = sorted(set(feature_ids) & set(label_ids))

    all_metrics = []
    for pid in tqdm(patient_ids, desc="evaluate"):
        feature_path = TEST_FEATURE_DIR / f"{pid}_profound.npy"
        image_path = TEST_IMAGE_DIR / f"{pid}.nii.gz"
        label_path = TEST_LABEL_DIR / f"{pid}.nii.gz"
        if not image_path.exists():
            image_path = TEST_IMAGE_DIR / f"{pid}_0000.nii.gz"

        feature_array = np.load(feature_path).astype(np.float32)
        feature_tensor = torch.from_numpy(feature_array).float().unsqueeze(0).to(device)
        if feature_tensor.dim() == 4: feature_tensor = feature_tensor.unsqueeze(0)

        with torch.inference_mode():
            logits = model(feature_tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        pred_mask = (probs > args.threshold).astype(np.uint8)

        image = sitk.ReadImage(str(image_path))
        pred_img = sitk.GetImageFromArray(pred_mask.astype(np.float32))
        pred_img.SetSpacing(image.GetSpacing())
        pred_img.SetOrigin(image.GetOrigin())
        pred_img.SetDirection(image.GetDirection())
        
        sitk.WriteImage(pred_img, str(PREDICTION_DIR / f"{pid}_advanced_pred.nii.gz"))

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
    
    report_path = REPORT_DIR / "phase3_advanced_baseline.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"metrics": mean_metrics, "per_case": all_metrics}, handle, indent=2)
    print(f"[eval] Results: Dice={mean_metrics['mean_dice']:.4f} | Prec={mean_metrics['mean_precision']:.4f} | Rec={mean_metrics['mean_recall']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced ProFound decoder experiments & sweep")
    parser.add_argument("--mode", choices=["train", "evaluate", "sweep"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=3.0) 
    parser.add_argument("--threshold", type=float, default=0.5)
    
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overfit-n", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_experiment(args)
    elif args.mode == "evaluate":
        evaluate_experiment(args)
    elif args.mode == "sweep":
        sweep_experiments(args)