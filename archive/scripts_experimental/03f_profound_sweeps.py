import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any
import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints" / "profound_baseline"
FEATURE_DIR = WORKSPACE_DIR / "data" / "02_frozen_features"
TRAIN_FEATURE_DIR = FEATURE_DIR / "train"
TRAIN_LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTr"

TARGET_SHAPE = (64, 160, 160)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resize_array_to_shape(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(array.shape) == target_shape: return array
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy()

class ProFoundDecoderDataset(Dataset):
    def __init__(self, feature_dir: Path, label_dir: Path, patient_ids: List[str]):
        self.patient_ids = patient_ids
        self.cache = {}
        for pid in tqdm(self.patient_ids, desc="Caching data to RAM", leave=False):
            feat = np.load(feature_dir / f"{pid}_profound.npy").astype(np.float32)
            label_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(label_dir / f"{pid}.nii.gz"))).astype(np.float32)
            label_arr = resize_array_to_shape((label_arr > 0).astype(np.float32), TARGET_SHAPE)
            self.cache[pid] = {
                "feature": torch.from_numpy(feat).float(),
                "mask": torch.from_numpy(label_arr).float().unsqueeze(0)
            }
    def __len__(self) -> int: return len(self.patient_ids)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: return self.cache[self.patient_ids[idx]]

class AdvancedProFoundDecoder3D(nn.Module):
    def __init__(self, in_channels: int = 768, base_channels: int = 64, dropout_p: float = 0.3):
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
    dims = tuple(range(1, logits.dim()))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean()

def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    loss = alpha * torch.pow(1.0 - p_t, gamma) * ce
    return loss.mean()

def run_experiment(trial_name: str, config: Dict[str, Any], train_dataset: Dataset, val_dataset: Dataset, device: torch.device):
    set_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = AdvancedProFoundDecoder3D(
        base_channels=config["base_channels"], 
        dropout_p=config["dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight = torch.tensor([config["pos_weight"]], device=device, dtype=torch.float32)

    best_val_dice = -1.0
    best_t = 0.5
    eval_thresholds = [0.55, 0.60, 0.65, 0.70]
    
    print(f"\n" + "="*80)
    print(f"🚀 {trial_name}")
    print(f"Config: Ch={config['base_channels']} | LR={config['lr']} | BS={config['batch_size']} | Drop={config['dropout']} | BCE={config['bce_weight']} | Dice={config['dice_weight']} | Focal={config['focal_weight']} | PosW={config['pos_weight']}")
    print("="*80)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch:02d}", leave=False):
            feat, mask = batch["feature"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            if feat.dim() == 4: feat = feat.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(scaler is not None)):
                logits = model(feat)
                loss = 0.0
                if config["bce_weight"] > 0: loss += config["bce_weight"] * nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)
                if config["dice_weight"] > 0: loss += config["dice_weight"] * soft_dice_loss(logits, mask)
                if config["focal_weight"] > 0: loss += config["focal_weight"] * focal_loss(logits, mask)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        
        model.eval()
        results = {t: [] for t in eval_thresholds}
        with torch.inference_mode():
            for batch in val_loader:
                feat, mask = batch["feature"].to(device), batch["mask"].to(device)
                if feat.dim() == 4: feat = feat.unsqueeze(0)
                if mask.dim() == 4: mask = mask.unsqueeze(0)
                
                probs = torch.sigmoid(model(feat))
                for t in eval_thresholds:
                    preds = (probs > t).float()
                    intersection = (preds * mask).sum().item()
                    denom = (preds.sum() + mask.sum() + 1e-6).item()
                    results[t].append((2.0 * intersection) / denom)
                    
        epoch_best_dice = 0.0
        epoch_best_t = 0.0
        for t in eval_thresholds:
            mean_dice = sum(results[t]) / len(results[t])
            if mean_dice > epoch_best_dice:
                epoch_best_dice = mean_dice
                epoch_best_t = t
                
        is_best = epoch_best_dice > best_val_dice
        if is_best:
            best_val_dice = epoch_best_dice
            best_t = epoch_best_t

        marker = "🔥 NEW BEST" if is_best else ""
        if epoch_best_dice >= 0.20:
            marker += " 🚀 (>0.20!)"
            
        print(f"Ep {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Best Dice: {epoch_best_dice:.4f} (at T={epoch_best_t:.2f}) {marker}")

    print(f"[{trial_name}] Finished! Absolute Peak: {best_val_dice:.4f} at Threshold {best_t:.2f}")
    return best_val_dice, best_t

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweeps] Using device: {device}")

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    pids = sorted(set(feature_ids) & set(label_ids))
    
    random.seed(42)
    random.shuffle(pids)
    train_ids, val_ids = pids[:int(len(pids)*0.8)], pids[int(len(pids)*0.8):]

    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids)
    val_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids)

    experiments = [
        # 1-3: The Original Triad
        {
            "name": "TRIAL 01: The Marathon (Stability Focus)",
            "config": {"epochs": 30, "lr": 1e-4, "batch_size": 8, "dropout": 0.3, "bce_weight": 0.5, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 3.0, "base_channels": 64}
        },
        {
            "name": "TRIAL 02: Pure Geometry (Focal + Dice Only)",
            "config": {"epochs": 30, "lr": 3e-4, "batch_size": 4, "dropout": 0.1, "bce_weight": 0.0, "dice_weight": 1.0, "focal_weight": 0.5, "pos_weight": 3.0, "base_channels": 64}
        },
        {
            "name": "TRIAL 03: The Edge-Cutter (Aggressive Weights)",
            "config": {"epochs": 30, "lr": 1e-4, "batch_size": 4, "dropout": 0.3, "bce_weight": 1.0, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 5.0, "base_channels": 64}
        },
        # 4-6: Network Capacity & Regularization
        {
            "name": "TRIAL 04: High Capacity, Low LR (The Deep Thinker)",
            "config": {"epochs": 30, "lr": 5e-5, "batch_size": 4, "dropout": 0.3, "bce_weight": 0.5, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 3.0, "base_channels": 128}
        },
        {
            "name": "TRIAL 05: The Extreme Regularizer (50% Dropout)",
            "config": {"epochs": 30, "lr": 3e-4, "batch_size": 8, "dropout": 0.5, "bce_weight": 0.5, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 3.0, "base_channels": 64}
        },
        {
            "name": "TRIAL 06: Pure Dice Overlap (No BCE, No Focal)",
            "config": {"epochs": 30, "lr": 1e-4, "batch_size": 4, "dropout": 0.1, "bce_weight": 0.0, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 1.0, "base_channels": 64}
        },
        # 7-10: Specialized Physics & Imbalance Handlers
        {
            "name": "TRIAL 07: The Kitchen Sink (BCE + Dice + Focal)",
            "config": {"epochs": 30, "lr": 2e-4, "batch_size": 4, "dropout": 0.2, "bce_weight": 0.5, "dice_weight": 1.0, "focal_weight": 0.5, "pos_weight": 3.0, "base_channels": 64}
        },
        {
            "name": "TRIAL 08: High Recall Pursuit (PosWeight 10.0)",
            "config": {"epochs": 30, "lr": 1e-4, "batch_size": 4, "dropout": 0.3, "bce_weight": 1.0, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 10.0, "base_channels": 64}
        },
        {
            "name": "TRIAL 09: The Micro-Batch (High Gradient Variance)",
            "config": {"epochs": 30, "lr": 5e-5, "batch_size": 2, "dropout": 0.1, "bce_weight": 0.5, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 3.0, "base_channels": 64}
        },
        {
            "name": "TRIAL 10: Cosine Baseline (The Control)",
            "config": {"epochs": 30, "lr": 3e-4, "batch_size": 4, "dropout": 0.1, "bce_weight": 0.5, "dice_weight": 1.0, "focal_weight": 0.0, "pos_weight": 3.0, "base_channels": 64}
        }
    ]

    summary = []
    for exp in experiments:
        best_dice, best_t = run_experiment(exp["name"], exp["config"], train_dataset, val_dataset, device)
        summary.append((exp["name"], best_dice, best_t))

    print("\n" + "="*80)
    print("🏆 FINAL 10-TRIAL SWEEP SUMMARY")
    print("="*80)
    for name, dice, t in summary:
        marker = " 🚀🚀🚀" if dice >= 0.20 else ""
        print(f"{name}: {dice:.4f} (at T={t:.2f}){marker}")
    print("="*80)

if __name__ == "__main__":
    main()