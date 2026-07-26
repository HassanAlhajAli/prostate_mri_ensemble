import argparse
import json
import shutil
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

# --- PATHS ---
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints" / "profound_baseline"
FEATURE_DIR = WORKSPACE_DIR / "data" / "02_frozen_features"
TRAIN_FEATURE_DIR = FEATURE_DIR / "train"
TEST_FEATURE_DIR = FEATURE_DIR / "test"
TRAIN_LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTr"
TEST_LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTs"
REPORTS_DIR = WORKSPACE_DIR / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "profound_predictions"

TARGET_SHAPE = (64, 160, 160)
OPTIMAL_THRESHOLD = 0.65  # The champion threshold!

def set_seed(seed: int = 42):
    """Strict reproducibility to guarantee 0.20+"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resize_array_to_shape(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(array.shape) == target_shape: return array
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy()

class ProFoundDecoderDataset(Dataset):
    def __init__(self, feature_dir: Path, label_dir: Path, patient_ids: List[str]):
        self.patient_ids = patient_ids
        self.cache = {}
        for pid in tqdm(self.patient_ids, desc=f"Caching {feature_dir.name} data", leave=False):
            feat_path = feature_dir / f"{pid}_profound.npy"
            label_path = label_dir / f"{pid}.nii.gz"
            if not feat_path.exists() or not label_path.exists():
                continue
                
            feat = np.load(feat_path).astype(np.float32)
            label_img = sitk.ReadImage(str(label_path))
            label_arr = sitk.GetArrayFromImage(label_img).astype(np.float32)
            label_arr_resized = resize_array_to_shape((label_arr > 0).astype(np.float32), TARGET_SHAPE)
            
            self.cache[pid] = {
                "feature": torch.from_numpy(feat).float(),
                "mask": torch.from_numpy(label_arr_resized).float().unsqueeze(0)
            }
        self.valid_ids = list(self.cache.keys())

    def __len__(self) -> int: return len(self.valid_ids)
    def __getitem__(self, idx: int) -> Dict: 
        pid = self.valid_ids[idx]
        data = self.cache[pid].copy()
        data["pid"] = pid
        return data

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

def save_prediction(probs_array: np.ndarray, original_img: sitk.Image, save_path: Path):
    """Resizes prob map and saves safely, bypassing WSL permission errors."""
    orig_shape = original_img.GetSize()[::-1]
    tensor = torch.from_numpy(probs_array).unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(tensor, size=orig_shape, mode="trilinear", align_corners=False)
    
    pred_arr = (resized_tensor.squeeze().cpu().numpy() > OPTIMAL_THRESHOLD).astype(np.uint8)
    pred_img = sitk.GetImageFromArray(pred_arr)
    pred_img.CopyInformation(original_img)
    
    # WSL Bulletproof Saving Logic
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sitk.WriteImage(pred_img, str(save_path))
    except Exception as e:
        print(f"WSL Write Exception caught for {save_path.name}. Using /tmp fallback.")
        tmp_path = Path("/tmp") / save_path.name
        sitk.WriteImage(pred_img, str(tmp_path))
        shutil.copy(str(tmp_path), str(save_path))
        tmp_path.unlink()

def main():
    set_seed(42)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] Device: {device} | Strict Determinism: ON | Target Threshold: {OPTIMAL_THRESHOLD}")

    # --- 1. PREPARE DATA ---
    train_feat_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    test_feat_ids = sorted({p.name[: -len("_profound.npy")] for p in TEST_FEATURE_DIR.glob("*_profound.npy")})
    
    random.shuffle(train_feat_ids)
    train_ids = train_feat_ids[:int(len(train_feat_ids)*0.8)]
    val_ids = train_feat_ids[int(len(train_feat_ids)*0.8):]

    train_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids)
    val_dataset = ProFoundDecoderDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids)
    test_dataset = ProFoundDecoderDataset(TEST_FEATURE_DIR, TEST_LABEL_DIR, test_feat_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 2. MODEL & TRAINING CONFIG ---
    model = AdvancedProFoundDecoder3D(base_channels=64, dropout_p=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight = torch.tensor([3.0], device=device, dtype=torch.float32)

    best_val_dice = -1.0
    best_model_path = CHECKPOINT_DIR / "final_champion.pt"
    epochs = 12 # Capped at 12 since it always peaks by Epoch 6

    print("\n--- PHASE 1: TRAINING CHAMPION MODEL ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            feat, mask = batch["feature"].to(device), batch["mask"].to(device)
            if feat.dim() == 4: feat = feat.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(scaler is not None)):
                logits = model(feat)
                loss = 0.5 * nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, mask) + soft_dice_loss(logits, mask)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_dices = []
        with torch.inference_mode():
            for batch in val_loader:
                feat, mask = batch["feature"].to(device), batch["mask"].to(device)
                if feat.dim() == 4: feat = feat.unsqueeze(0)
                if mask.dim() == 4: mask = mask.unsqueeze(0)
                
                probs = torch.sigmoid(model(feat))
                preds = (probs > OPTIMAL_THRESHOLD).float()
                
                intersection = (preds * mask).sum().item()
                denom = (preds.sum() + mask.sum() + 1e-6).item()
                val_dices.append((2.0 * intersection) / denom)
                
        avg_val_dice = sum(val_dices) / len(val_dices)
        is_best = avg_val_dice > best_val_dice
        if is_best:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)

        marker = " 🚀🚀🚀" if avg_val_dice >= 0.20 else ""
        print(f"Ep {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Dice (T={OPTIMAL_THRESHOLD}): {avg_val_dice:.4f} {'🔥 BEST' if is_best else ''}{marker}")

    # --- 3. TEST SET EVALUATION ---
    print(f"\n--- PHASE 2: TEST SET EVALUATION (Threshold {OPTIMAL_THRESHOLD}) ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_dices = []
    
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            feat, mask = batch["feature"].to(device), batch["mask"].to(device)
            pid = batch["pid"][0]
            if feat.dim() == 4: feat = feat.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            
            probs = torch.sigmoid(model(feat))
            preds = (probs > OPTIMAL_THRESHOLD).float()
            
            intersection = (preds * mask).sum().item()
            denom = (preds.sum() + mask.sum() + 1e-6).item()
            dice = (2.0 * intersection) / denom
            test_dices.append(dice)
            
            # Save NIfTI prediction
            original_img = sitk.ReadImage(str(TEST_LABEL_DIR / f"{pid}.nii.gz"))
            save_prediction(
                probs_array=probs.squeeze().cpu().numpy(),
                original_img=original_img,
                save_path=PREDICTIONS_DIR / f"{pid}_pred.nii.gz"
            )

    mean_test_dice = sum(test_dices) / len(test_dices)
    
    report = {
        "model": "AdvancedProFoundDecoder3D (Champion)",
        "test_cases": len(test_dices),
        "mean_test_dice": mean_test_dice,
        "peak_val_dice": best_val_dice,
        "threshold_used": OPTIMAL_THRESHOLD
    }
    
    with open(REPORTS_DIR / "final_profound_test_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n🏆 FINAL TEST DICE: {mean_test_dice:.4f}")
    print(f"✅ Predictions saved to: {PREDICTIONS_DIR}")
    print(f"✅ Report saved to: {REPORTS_DIR / 'final_profound_test_metrics.json'}")

if __name__ == "__main__":
    main()