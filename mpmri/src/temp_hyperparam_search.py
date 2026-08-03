import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List, Dict
import random
import gc
from tqdm import tqdm

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
MPMRI_DIR = SCRIPT_DIR.parent
ROOT_DIR = MPMRI_DIR.parent

FEATURE_DIR = MPMRI_DIR / "data" / "03_frozen_features_mpmri"
TRAIN_FEATURE_DIR = FEATURE_DIR / "train"
TRAIN_LABEL_DIR = MPMRI_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset501_ProstateMPMRI" / "labelsTr"

TARGET_SHAPE = (64, 160, 160)

def set_seed(seed: int = 42):
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

class FastValidationDataset(Dataset):
    def __init__(self, feature_dir: Path, label_dir: Path, patient_ids: List[str]):
        self.patient_ids = patient_ids
        self.cache = {}
        for pid in tqdm(self.patient_ids, desc=f"Loading data to RAM"):
            feat_path = feature_dir / f"{pid}_profound.npy"
            label_path = label_dir / f"{pid}.nii.gz"
            if not feat_path.exists() or not label_path.exists(): continue
                
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
        return self.cache[self.valid_ids[idx]].copy()

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

def run_experiment(exp_name, lr, pos_weight, dropout, epochs, train_loader, val_loader, device):
    model = AdvancedProFoundDecoder3D(base_channels=64, dropout_p=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    best_val_dice = -1.0

    print(f"\n🚀 Running {exp_name} [LR: {lr}, Pos: {pos_weight}, Drop: {dropout}, Ep: {epochs}]")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            feat, mask = batch["feature"].to(device), batch["mask"].to(device)
            if feat.dim() == 4: feat = feat.unsqueeze(0)
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(scaler is not None)):
                logits = model(feat)
                loss = 0.5 * nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)(logits, mask) + soft_dice_loss(logits, mask)
            
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
                preds = (probs > 0.50).float()
                
                intersection = (preds * mask).sum().item()
                denom = (preds.sum() + mask.sum() + 1e-6).item()
                val_dices.append((2.0 * intersection) / denom)
                
        avg_val_dice = sum(val_dices) / len(val_dices)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice

        # PRINTING EVERY EPOCH
        print(f"   ↳ Epoch {epoch:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f} (Peak: {best_val_dice:.4f})")

    # Free up memory before the next experiment
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_dice

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*50)
    print(f"[*] 🖥️  HARDWARE STATUS")
    if device.type == 'cuda':
        print(f"    ↳ GPU Accelerated = TRUE")
        print(f"    ↳ GPU Target      = {torch.cuda.get_device_name(0)}")
    else:
        print(f"    ↳ GPU Accelerated = FALSE (Running on CPU - This will be slow!)")
    print("="*50 + "\n")

    # Load Train/Val data purely into memory
    print("[*] 💾 COMMENCING IN-MEMORY DATA CACHING...")
    train_feat_ids = sorted({p.name[: -len("_profound.npy")] for p in TRAIN_FEATURE_DIR.glob("*_profound.npy")})
    random.shuffle(train_feat_ids)
    
    split_idx = int(len(train_feat_ids) * 0.8)
    train_ids = train_feat_ids[:split_idx]
    val_ids = train_feat_ids[split_idx:]

    train_dataset = FastValidationDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, train_ids)
    val_dataset = FastValidationDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, val_ids)
    
    print(f"\n[*] ✅ CACHING COMPLETE!")
    print(f"    ↳ {len(train_dataset)} Train volumes locked in RAM.")
    print(f"    ↳ {len(val_dataset)} Validation volumes locked in RAM.")
    print(f"    ↳ Zero Hard Drive I/O will be used during training.\n")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # The 10 Experiments Grid
    experiments = [
        {"name": "Exp_01_Baseline", "lr": 3e-4, "pos": 5.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_02_High_LR", "lr": 1e-3, "pos": 5.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_03_Low_LR", "lr": 1e-4, "pos": 5.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_04_High_Pos7", "lr": 3e-4, "pos": 7.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_05_Extreme_Pos10", "lr": 3e-4, "pos": 10.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_06_Low_Pos3", "lr": 3e-4, "pos": 3.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_07_High_Dropout", "lr": 3e-4, "pos": 5.0, "drop": 0.3, "ep": 20},
        {"name": "Exp_08_Long_Train", "lr": 3e-4, "pos": 5.0, "drop": 0.1, "ep": 30},
        {"name": "Exp_09_Agro_Combo", "lr": 5e-4, "pos": 7.0, "drop": 0.1, "ep": 20},
        {"name": "Exp_10_Safe_Combo", "lr": 1e-4, "pos": 5.0, "drop": 0.2, "ep": 30},
    ]

    results = []
    
    for exp in experiments:
        score = run_experiment(
            exp["name"], exp["lr"], exp["pos"], exp["drop"], exp["ep"], 
            train_loader, val_loader, device
        )
        results.append((exp["name"], score, exp))

    # Print Leaderboard
    print("\n" + "="*50)
    print("🏆 HYPERPARAMETER SEARCH LEADERBOARD 🏆")
    print("="*50)
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, score, config) in enumerate(results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} #{rank}: {name} - Peak Val Dice: {score:.4f}")
        print(f"       (LR: {config['lr']}, Pos: {config['pos']}, Drop: {config['drop']}, Epochs: {config['ep']})")
    print("="*50)

if __name__ == "__main__":
    main()