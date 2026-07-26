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
    def __init__(self, in_channels: int = 768, base_channels: int = 64, dropout_p: float = 0.1):
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
    best_ckpt_path = CHECKPOINT_DIR / "champion_01956.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train-sweep] Using device: {device}")

    # Patient split matching previous runs
    feature_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    label_ids = sorted({p.name[: -len(".nii.gz")] for p in TRAIN_LABEL_DIR.glob("*.nii.gz")})
    pids = sorted(set(feature_ids) & set(label_ids))
    
    random.seed(42)
    random.shuffle(pids)
    train_ids, val_ids = pids[:int(len(pids)*0.8)], pids[int(len(pids)*0.8):]
    print(f"[train-sweep] Train cases: {len(train_ids)} | Val cases: {len(val_ids)}")

    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids)
    val_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = AdvancedProFoundDecoder3D(base_channels=64, dropout_p=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight = torch.tensor([3.0], device=device, dtype=torch.float32)

    best_val_dice = -1.0
    epochs = 20  # More than enough since peak happens around epoch 5-6

    print("\n--- STARTING TRAINING (0.1956 Winning Config) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False):
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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        # Validation evaluation at default threshold 0.5
        model.eval()
        val_dices = []
        with torch.inference_mode():
            for batch in val_loader:
                feat, mask = batch["feature"].to(device), batch["mask"].to(device)
                if feat.dim() == 4: feat = feat.unsqueeze(0)
                if mask.dim() == 4: mask = mask.unsqueeze(0)
                probs = torch.sigmoid(model(feat))
                preds = (probs > 0.5).float()
                intersection = (preds * mask).sum().item()
                denom = (preds.sum() + mask.sum() + 1e-6).item()
                val_dices.append((2.0 * intersection) / denom)
                
        avg_val_dice = sum(val_dices) / len(val_dices)
        is_best = avg_val_dice > best_val_dice
        if is_best:
            best_val_dice = avg_val_dice
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt_path)

        print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Dice (T=0.5): {avg_val_dice:.4f} {'🔥 NEW BEST' if is_best else ''}")

    print(f"\n[train-sweep] Training finished! Best checkpoint saved to: {best_ckpt_path}")

    # --- RUN SYSTEMATIC THRESHOLD SWEEP ---
    print("\n" + "="*40)
    print("      RUNNING SYSTEMATIC THRESHOLD SWEEP      ")
    print("="*40)
    
    # Load best weights back into model
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    results = {t: [] for t in thresholds}

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Sweeping thresholds"):
            feat, mask = batch["feature"].to(device), batch["mask"].to(device)
            if feat.dim() == 4: feat = feat.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            
            probs = torch.sigmoid(model(feat))
            
            for t in thresholds:
                preds = (probs > t).float()
                intersection = (preds * mask).sum().item()
                denom = (preds.sum() + mask.sum() + 1e-6).item()
                dice = (2.0 * intersection) / denom
                results[t].append(dice)

    best_t = 0.0
    best_score = 0.0
    for t in thresholds:
        mean_dice = sum(results[t]) / len(results[t])
        marker = ""
        if mean_dice > best_score:
            best_score = mean_dice
            best_t = t
        if mean_dice >= 0.20:
            marker = " 🚀 (OVER 0.20!)"
        print(f" Threshold {t:.2f}  ->  Val Dice: {mean_dice:.4f}{marker}")
        
    print("="*40)
    print(f"🏆 OPTIMAL THRESHOLD: {best_t:.2f} with Dice: {best_score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()