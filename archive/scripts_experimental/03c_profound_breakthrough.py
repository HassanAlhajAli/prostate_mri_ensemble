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
from scipy.ndimage import label as scipy_label
from scipy.ndimage import sum as scipy_sum

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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. DATASET WITH CACHING & 3D SPATIAL AUGMENTATION
# ==============================================================================
def resize_array_to_shape(array: np.ndarray, target_shape: Tuple[int, int, int], mode: str = "nearest") -> np.ndarray:
    if tuple(array.shape) == target_shape:
        return array
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False if mode == "trilinear" else None)
    return resized.squeeze(0).squeeze(0).cpu().numpy()

class ProFoundBreakthroughDataset(Dataset):
    def __init__(self, feature_dir: Path, label_dir: Path, patient_ids: List[str], use_cache: bool = True, augment: bool = False):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.patient_ids = patient_ids
        self.use_cache = use_cache
        self.augment = augment
        self.cache = {}
        
        if self.use_cache:
            for pid in tqdm(self.patient_ids, desc="Caching dataset to RAM", leave=False):
                self.cache[pid] = self._load_from_disk(pid)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_from_disk(self, pid: str) -> Dict[str, Any]:
        feature_path = self.feature_dir / f"{pid}_profound.npy"
        label_path = self.label_dir / f"{pid}.nii.gz"
        
        feature = np.load(feature_path).astype(np.float32)
        label_image = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label_image).astype(np.float32)
        label_array = (label_array > 0).astype(np.float32)

        if tuple(label_array.shape) != TARGET_SHAPE:
            label_array = resize_array_to_shape(label_array, TARGET_SHAPE, mode="nearest")

        return {
            "patient_id": pid, 
            "feature": torch.from_numpy(feature).float(), 
            "mask": torch.from_numpy(label_array).float().unsqueeze(0)
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.patient_ids[idx]
        data = self.cache[pid] if self.use_cache else self._load_from_disk(pid)
        
        # Clone tensors so augmentations don't poison the RAM cache permanently
        feat = data["feature"].clone()
        mask = data["mask"].clone()
        
        # Synchronized 3D Spatial Flips
        if self.augment:
            for dim in [-1, -2, -3]: # X, Y, Z axes
                if random.random() > 0.5:
                    feat = torch.flip(feat, dims=[dim])
                    mask = torch.flip(mask, dims=[dim])

        return {"patient_id": pid, "feature": feat, "mask": mask}

# ==============================================================================
# 3. POST-PROCESSING & METRICS
# ==============================================================================
def remove_small_connected_components(binary_mask: np.ndarray, min_size: int = 15) -> np.ndarray:
    """Removes impossible tiny noise clusters from predictions."""
    labeled_mask, num_features = scipy_label(binary_mask)
    if num_features == 0:
        return binary_mask
    component_sizes = scipy_sum(binary_mask, labeled_mask, range(1, num_features + 1))
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_mask - 1]
    too_small_mask[labeled_mask == 0] = False
    binary_mask[too_small_mask] = 0
    return binary_mask

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
# 4. SQUEEZE-AND-EXCITATION DECODER WITH DEEP SUPERVISION
# ==============================================================================
class SEResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        # Squeeze-and-Excitation Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, max(1, out_channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, out_channels // 4), out_channels, 1),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout3d(p=dropout_p)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels)
        ) if in_channels != out_channels else nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.leaky_relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = out * self.se(out) # Apply Channel Attention
        out += res
        return F.leaky_relu(out, inplace=True)

class BreakthroughDecoder3D(nn.Module):
    def __init__(self, in_channels: int = 768, base_channels: int = 64, dropout_p: float = 0.2):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, base_channels, kernel_size=1)
        
        self.layer1 = SEResidualBlock3D(base_channels, base_channels // 2, dropout_p)
        self.layer2 = SEResidualBlock3D(base_channels // 2, base_channels // 4, dropout_p)
        self.layer3 = SEResidualBlock3D(base_channels // 4, base_channels // 8, dropout_p)
        self.layer4 = SEResidualBlock3D(base_channels // 8, base_channels // 16, dropout_p)
        
        # Deep Supervision Heads
        self.head_coarse = nn.Conv3d(base_channels // 4, 1, kernel_size=1)
        self.head_mid = nn.Conv3d(base_channels // 8, 1, kernel_size=1)
        self.head_final = nn.Conv3d(base_channels // 16, 1, kernel_size=1)
        
        nn.init.constant_(self.head_coarse.bias, -2.5)
        nn.init.constant_(self.head_mid.bias, -2.5)
        nn.init.constant_(self.head_final.bias, -2.5)

    def forward(self, x: torch.Tensor):
        x0 = self.proj(x)
        x1 = self.layer1(F.interpolate(x0, scale_factor=2, mode="trilinear", align_corners=False))
        x2 = self.layer2(F.interpolate(x1, scale_factor=2, mode="trilinear", align_corners=False))
        out_coarse = self.head_coarse(x2)
        
        x3 = self.layer3(F.interpolate(x2, scale_factor=2, mode="trilinear", align_corners=False))
        out_mid = self.head_mid(x3)
        
        x4 = self.layer4(F.interpolate(x3, scale_factor=2, mode="trilinear", align_corners=False))
        out_final = self.head_final(x4)
        
        if self.training:
            return out_final, out_mid, out_coarse
        return out_final

# ==============================================================================
# 5. FOCAL TVERSKY LOSS
# ==============================================================================
def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, eps: float = 1e-6) -> torch.Tensor:
    """Alpha controls FP penalty. Beta controls FN penalty. Beta > Alpha forces model to penalize missing lesion pixels heavily."""
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, logits.dim()))

    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = torch.pow(1.0 - tversky, gamma)
    return loss.mean()

# ==============================================================================
# 6. SWEEP RUNNER
# ==============================================================================
def run_breakthrough_experiment(trial_id: int, config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> float:
    set_seed(config["seed"])
    model = BreakthroughDecoder3D(base_channels=config["decoder_channels"], dropout_p=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    best_val_dice = -1.0
    trial_checkpoint_path = CHECKPOINT_DIR / f"trial_bt_{trial_id:02d}_best.pt"

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        
        batch_iterator = tqdm(train_loader, desc=f"Trial {trial_id:02d} | Ep {epoch:02d}/{config['epochs']}", leave=False)
        for batch in batch_iterator:
            feature = batch["feature"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if feature.dim() == 4: feature = feature.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(scaler is not None)):
                # Deep Supervision outputs
                logits_final, logits_mid, logits_coarse = model(feature)
                
                # Resize intermediates to match mask shape
                logits_mid = F.interpolate(logits_mid, size=mask.shape[2:], mode="trilinear", align_corners=False)
                logits_coarse = F.interpolate(logits_coarse, size=mask.shape[2:], mode="trilinear", align_corners=False)
                
                # Combined Loss
                loss_f = focal_tversky_loss(logits_final, mask)
                loss_m = focal_tversky_loss(logits_mid, mask)
                loss_c = focal_tversky_loss(logits_coarse, mask)
                loss = loss_f + 0.5 * loss_m + 0.25 * loss_c

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))

        # Validation Loop with Connected Component cleanup
        model.eval()
        val_dices = []
        with torch.inference_mode():
            for batch in val_loader:
                feature = batch["feature"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                if feature.dim() == 4: feature = feature.unsqueeze(0)
                if mask.dim() == 4: mask = mask.unsqueeze(0)
                
                logits = model(feature) # In eval mode, it only returns out_final
                preds_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # Threshold and Cleanup Noise
                preds_bin = (preds_prob > config["threshold"]).astype(np.uint8)
                preds_clean = remove_small_connected_components(preds_bin, min_size=15)
                
                preds_tensor = torch.from_numpy(preds_clean).to(device).unsqueeze(0).unsqueeze(0)
                
                intersection = (preds_tensor * mask).sum()
                denom = preds_tensor.sum() + mask.sum()
                dice_score = (2.0 * intersection) / (denom + 1e-6)
                val_dices.append(dice_score.item())

        avg_val_dice = sum(val_dices) / max(1, len(val_dices))
        is_best = avg_val_dice > best_val_dice
        if is_best:
            best_val_dice = avg_val_dice
            torch.save({"model_state_dict": model.state_dict(), "config": config, "val_dice": best_val_dice}, trial_checkpoint_path)

        print(f"  [Trial {trial_id:02d}] Epoch {epoch:02d}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}{' (New Best!)' if is_best else ''}")

    print(f"[Trial {trial_id:02d}] Finished | Best Val Dice: {best_val_dice:.4f}\n")
    return best_val_dice

def sweep_experiments(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] Starting 8-Trial Breakthrough Sweep on device={device}")

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    patient_ids = sorted(set(feature_ids) & set(label_ids))

    random.seed(args.seed)
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)
    train_ids, val_ids = patient_ids[:split_idx], patient_ids[split_idx:]

    # Train dataset has augment=True to apply 3D flips!
    train_dataset = ProFoundBreakthroughDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids, use_cache=True, augment=True)
    val_dataset = ProFoundBreakthroughDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids, use_cache=True, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 8 Targeted Experiments (Locking channels to 64 since we know 32 fails)
    search_space = [
        {"lr": 1e-4, "decoder_channels": 64, "dropout": 0.2, "epochs": 50, "threshold": 0.35, "seed": 42},
        {"lr": 1e-4, "decoder_channels": 64, "dropout": 0.3, "epochs": 50, "threshold": 0.35, "seed": 42},
        {"lr": 2e-4, "decoder_channels": 64, "dropout": 0.2, "epochs": 50, "threshold": 0.35, "seed": 42},
        {"lr": 2e-4, "decoder_channels": 64, "dropout": 0.3, "epochs": 50, "threshold": 0.35, "seed": 42},
        {"lr": 1e-4, "decoder_channels": 64, "dropout": 0.2, "epochs": 50, "threshold": 0.40, "seed": 42},
        {"lr": 1e-4, "decoder_channels": 64, "dropout": 0.3, "epochs": 50, "threshold": 0.40, "seed": 42},
        {"lr": 2e-4, "decoder_channels": 64, "dropout": 0.2, "epochs": 50, "threshold": 0.40, "seed": 42},
        {"lr": 2e-4, "decoder_channels": 64, "dropout": 0.3, "epochs": 50, "threshold": 0.40, "seed": 42},
    ]

    results = []
    best_overall_dice = -1.0
    best_config = None

    for i, config in enumerate(search_space):
        trial_id = i + 1
        print(f"\n--- Running Trial {trial_id}/8 with config: {config} ---")
        val_dice = run_breakthrough_experiment(trial_id, config, train_loader, val_loader, device)
        results.append({"trial_id": trial_id, "val_dice": val_dice, "config": config})
        
        if val_dice > best_overall_dice:
            best_overall_dice = val_dice
            best_config = config
            src_path = CHECKPOINT_DIR / f"trial_bt_{trial_id:02d}_best.pt"
            dst_path = CHECKPOINT_DIR / "best_breakthrough.pt"
            if src_path.exists():
                torch.save(torch.load(src_path, weights_only=False), dst_path)

    summary_path = REPORT_DIR / "phase3_breakthrough_sweep.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_dice": best_overall_dice, "best_config": best_config, "trials": results}, f, indent=2)

    print("\n==============================================")
    print(f" BREAKTHROUGH SWEEP COMPLETE! Best Val Dice: {best_overall_dice:.4f}")
    print(f" Best Config: {best_config}")
    print("==============================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breakthrough ProFound experiments")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    
    sweep_experiments(args)