import argparse
from pathlib import Path
from typing import Tuple, List, Dict
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

def main():
    set_seed(42)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = CHECKPOINT_DIR / "slow_burn_champion.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[slow-burn] Device: {device}")

    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    pids = sorted(set(feature_ids) & set(label_ids))
    
    random.seed(42)
    random.shuffle(pids)
    train_ids, val_ids = pids[:int(len(pids)*0.8)], pids[int(len(pids)*0.8):]

    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids)
    val_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids)
    
    # BS=8 to ensure smooth gradients
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = AdvancedProFoundDecoder3D(base_channels=64, dropout_p=0.3).to(device)
    
    # Heavy weight decay (1e-2) to prevent early memorization
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    
    epochs = 100
    
    # OneCycleLR automatically warms up the LR for the first 20% of training, then decays
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-5, 
        epochs=epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.2, # Peaks at epoch 20
        div_factor=25.0, # Starts very slow
        final_div_factor=1000.0 # Ends near zero
    )
    
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight = torch.tensor([3.0], device=device, dtype=torch.float32)

    best_val_dice = -1.0
    eval_thresholds = [0.55, 0.60, 0.65, 0.70]

    print("\n" + "="*80)
    print("🔥 THE 100-EPOCH SLOW BURN")
    print("Config: Ch=64 | Max LR=5e-5 | BS=8 | Drop=0.3 | Warmup+Decay | Grad Clip=1.0")
    print("="*80)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch:03d}/{epochs}", leave=False):
            feat, mask = batch["feature"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            if feat.dim() == 4: feat = feat.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(scaler is not None)):
                logits = model(feat)
                bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask)
                dice = soft_dice_loss(logits, mask)
                loss = 0.5 * bce + 1.0 * dice
            
            if scaler:
                scaler.scale(loss).backward()
                # Unscale before clipping
                scaler.unscale_(optimizer)
                # GRADIENT CLIPPING: Prevents massive leaps in loss landscape
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            train_loss += loss.item()
            # Step the scheduler per batch for smooth warmup
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        # Multi-Threshold Validation
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
            torch.save({"model_state_dict": model.state_dict(), "best_t": epoch_best_t}, best_ckpt_path)

        marker = "🔥 NEW BEST" if is_best else ""
        if epoch_best_dice >= 0.20:
            marker += " 🚀 (>0.20!)"

        # Only print every epoch if it's a new best, otherwise print every 5th epoch to keep logs clean
        if is_best or epoch % 5 == 0:
            print(f"Ep {epoch:03d} | LR: {current_lr:.6f} | Loss: {train_loss/len(train_loader):.4f} | Best Dice: {epoch_best_dice:.4f} (at T={epoch_best_t:.2f}) {marker}")

    print(f"\n[slow-burn] Training complete! Absolute Peak Dice: {best_val_dice:.4f}")
    print(f"[slow-burn] Saved to: {best_ckpt_path}")

if __name__ == "__main__":
    main()