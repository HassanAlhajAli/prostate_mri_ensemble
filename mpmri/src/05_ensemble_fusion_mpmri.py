import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Safely handle numpy globals without triggering deprecation warnings
try:
    safe_globals = [np.dtype, np.bool_, np.int64, np.int32, np.float32, np.float64]
    if hasattr(np, 'core'):
        safe_globals.append(np.core.multiarray.scalar)
    if hasattr(np, '_core'):
        safe_globals.append(np._core.multiarray.scalar)
    torch.serialization.add_safe_globals(safe_globals)
except Exception:
    pass

_original_torch_load = torch.load

def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _compat_torch_load

# --- PATHS (Encapsulated in mpmri) ---
SCRIPT_DIR = Path(__file__).resolve().parent
MPMRI_DIR = SCRIPT_DIR.parent
ROOT_DIR = MPMRI_DIR.parent

DATASET_DIR = MPMRI_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset501_ProstateMPMRI"
TRAIN_IMAGE_DIR = DATASET_DIR / "imagesTr"
TRAIN_LABEL_DIR = DATASET_DIR / "labelsTr"
TEST_IMAGE_DIR = DATASET_DIR / "imagesTs"
TEST_LABEL_DIR = DATASET_DIR / "labelsTs"

# CORRECTED: Point directly to your custom nnU-Net checkpoint folder!
NNUNET_MODEL_DIR = MPMRI_DIR / "checkpoints" / "nnunet_mpmri"
NNUNET_CHECKPOINT_NAME = "checkpoint_final.pth"

PROFOUND_FEATURE_DIR = MPMRI_DIR / "data" / "03_frozen_features_mpmri"
PROFOUND_TRAIN_FEATURE_DIR = PROFOUND_FEATURE_DIR / "train"
PROFOUND_TEST_FEATURE_DIR = PROFOUND_FEATURE_DIR / "test"

# Directly point to the champion we just trained
PROFOUND_CHAMPION_PATH = MPMRI_DIR / "checkpoints" / "profound_mpmri" / "final_mpmri_champion.pt"

REPORT_DIR = MPMRI_DIR / "reports"
ENSEMBLE_PRED_DIR = REPORT_DIR / "ensemble_predictions_mpmri"
ENSEMBLE_CACHE_DIR = REPORT_DIR / "ensemble_cache_mpmri"
METRICS_PATH = REPORT_DIR / "ensemble_metrics_mpmri.json"

PHASE4_CHECKPOINT_DIR = MPMRI_DIR / "checkpoints" / "lomix_fusion_mpmri"
LOMIX_CHECKPOINT_PATH = PHASE4_CHECKPOINT_DIR / "lomix_best.pt"

TARGET_SHAPE = (64, 160, 160)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in [REPORT_DIR, ENSEMBLE_PRED_DIR, ENSEMBLE_CACHE_DIR, PHASE4_CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def case_id_from_image_path(path: Path) -> str:
    if path.name.endswith("_0000.nii.gz"):
        return path.name[: -len("_0000.nii.gz")]
    return path.name.replace(".nii.gz", "")


def list_case_ids(image_dir: Path) -> List[str]:
    return sorted(case_id_from_image_path(path) for path in image_dir.glob("*_0000.nii.gz"))


def read_nifti(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return array, image


def write_mask(mask_array: np.ndarray, reference_image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_image.CopyInformation(reference_image)
    try:
        sitk.WriteImage(mask_image, str(path))
    except Exception:
        # WSL fallback
        tmp = Path("/tmp") / path.name
        sitk.WriteImage(mask_image, str(tmp))
        shutil.copy2(str(tmp), str(path))
        tmp.unlink()


def write_probability_alias(prob_array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prob_array.astype(np.float32))


def load_probability_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".npz":
        archive = np.load(path)
        for key in ["probabilities", "softmax", "prob", "probs"]:
            if key in archive:
                return archive[key].astype(np.float32)
        return archive[archive.files[0]].astype(np.float32)
    raise ValueError(f"Unsupported probability file: {path}")


def normalize_probability_volume(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 4 and prob.shape[0] >= 2:
        return prob[1]
    if prob.ndim == 4 and prob.shape[0] == 1:
        return prob[0]
    if prob.ndim == 3:
        return prob
    raise ValueError(f"Unexpected probability array shape: {prob.shape}")


def threshold_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(np.uint8)


def dice_from_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = (pred > 0).astype(np.uint8)
    target_b = (target > 0).astype(np.uint8)
    inter = int(np.sum(pred_b & target_b))
    denom = int(np.sum(pred_b) + np.sum(target_b))
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def precision_recall_from_binary(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    pred_b = (pred > 0).astype(np.uint8)
    target_b = (target > 0).astype(np.uint8)
    tp = int(np.sum((pred_b == 1) & (target_b == 1)))
    fp = int(np.sum((pred_b == 1) & (target_b == 0)))
    fn = int(np.sum((pred_b == 0) & (target_b == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return float(precision), float(recall)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    dice = dice_from_binary(pred, target)
    precision, recall = precision_recall_from_binary(pred, target)
    return {"dice": dice, "precision": precision, "recall": recall}


def split_train_val(case_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    case_ids = sorted(case_ids)
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    val_count = max(1, int(len(case_ids) * val_ratio)) if len(case_ids) > 1 else 0
    val_ids = list(case_ids[:val_count])
    train_ids = list(case_ids[val_count:])
    if not train_ids:
        train_ids = list(val_ids)
    return train_ids, val_ids


class NnUNetProbabilitySource:
    def __init__(self, checkpoint_dir: Path, checkpoint_name: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.predictor = None

    def _build_predictor(self):
        if self.predictor is not None:
            return self.predictor
        
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Cannot find nnU-Net model at {self.checkpoint_dir}. Check your folder structure!")
            
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(self.checkpoint_dir),
            use_folds=(0,),
            checkpoint_name=self.checkpoint_name,
        )
        self.predictor = predictor
        return predictor

    def generate_split(self, image_dir: Path, split_dir: Path, force: bool = False) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        # Only grab _0000.nii.gz to identify unique patients
        t2_files = sorted(image_dir.glob("*_0000.nii.gz"))
        if not t2_files:
            raise RuntimeError(f"No images found in {image_dir}")

        expected = [split_dir / f"{case_id_from_image_path(path)}_prob.npy" for path in t2_files]
        if not force and all(path.exists() for path in expected):
            return

        # MAJOR FIX FOR MPMRI: nnUNet needs all 3 channels stacked per patient
        list_of_lists = []
        for t2_path in t2_files:
            case_id = case_id_from_image_path(t2_path)
            adc_path = t2_path.parent / f"{case_id}_0001.nii.gz"
            dwi_path = t2_path.parent / f"{case_id}_0002.nii.gz"
            
            if not adc_path.exists() or not dwi_path.exists():
                raise FileNotFoundError(f"Missing ADC or DWI for {case_id}")
                
            list_of_lists.append([str(t2_path), str(adc_path), str(dwi_path)])

        predictor = self._build_predictor()
        predictor.predict_from_files(
            list_of_lists,
            str(split_dir),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        for t2_path in t2_files:
            case_id = case_id_from_image_path(t2_path)
            seg_path = split_dir / f"{case_id}.nii.gz"
            prob_path = split_dir / f"{case_id}.npz"
            if prob_path.exists():
                prob_array = normalize_probability_volume(load_probability_file(prob_path))
            else:
                seg_array, _ = read_nifti(seg_path)
                prob_array = seg_array.astype(np.float32)
            write_probability_alias(prob_array, split_dir / f"{case_id}_prob.npy")
            if seg_path.exists():
                shutil.copy2(seg_path, split_dir / f"{case_id}_mask.nii.gz")


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

    def _make_block(self, in_channels: int, out_channels: int, dropout_p: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.conv1(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv2(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv3(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv4(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        return self.out_head(x)


def load_profound_model(device: torch.device) -> nn.Module:
    if not PROFOUND_CHAMPION_PATH.exists():
        raise FileNotFoundError(f"Missing ProFound checkpoint at: {PROFOUND_CHAMPION_PATH}")
    model = AdvancedProFoundDecoder3D(in_channels=768, base_channels=64, dropout_p=0.1)
    state = torch.load(PROFOUND_CHAMPION_PATH, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def profound_feature_path(split: str, case_id: str) -> Path:
    feature_dir = PROFOUND_TRAIN_FEATURE_DIR if split == "train" else PROFOUND_TEST_FEATURE_DIR
    return feature_dir / f"{case_id}_profound.npy"


def profound_image_path(split: str, case_id: str) -> Path:
    image_dir = TRAIN_IMAGE_DIR if split == "train" else TEST_IMAGE_DIR
    return image_dir / f"{case_id}_0000.nii.gz"


def profound_label_path(split: str, case_id: str) -> Path:
    label_dir = TRAIN_LABEL_DIR if split == "train" else TEST_LABEL_DIR
    return label_dir / f"{case_id}.nii.gz"


def generate_profound_split(split: str, case_ids: Sequence[str], model: nn.Module, device: torch.device, force: bool = False) -> None:
    split_dir = ENSEMBLE_CACHE_DIR / split / "profound"
    split_dir.mkdir(parents=True, exist_ok=True)
    expected = [split_dir / f"{case_id}_prob.npy" for case_id in case_ids]
    if not force and all(path.exists() for path in expected):
        return

    for case_id in tqdm(case_ids, desc=f"profound-{split}"):
        prob_path = split_dir / f"{case_id}_prob.npy"
        mask_path = split_dir / f"{case_id}_mask.nii.gz"
        if not force and prob_path.exists() and mask_path.exists():
            continue
        feature_path = profound_feature_path(split, case_id)
        image_path = profound_image_path(split, case_id)
        if not feature_path.exists() or not image_path.exists():
            raise FileNotFoundError(f"Missing ProFound input for {case_id} in {split}")
        feature = np.load(feature_path).astype(np.float32)
        feature_tensor = torch.from_numpy(feature).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(feature_tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        write_probability_alias(probs, prob_path)
        image = sitk.ReadImage(str(image_path))
        write_mask((probs >= 0.5).astype(np.uint8), image, mask_path)


def load_binary_mask(path: Path, fallback_prob: Optional[np.ndarray] = None) -> np.ndarray:
    try:
        array, _ = read_nifti(path)
        return (array > 0).astype(np.uint8)
    except Exception:
        if fallback_prob is not None:
            return (np.asarray(fallback_prob) > 0.5).astype(np.uint8)
        raise


def load_base_bundle(split: str, case_id: str) -> Dict[str, np.ndarray]:
    nnunet_dir = ENSEMBLE_CACHE_DIR / split / "nnunet"
    profound_dir = ENSEMBLE_CACHE_DIR / split / "profound"
    nnunet_prob = load_probability_file(nnunet_dir / f"{case_id}_prob.npy")
    profound_prob = load_probability_file(profound_dir / f"{case_id}_prob.npy")
    nnunet_mask = load_binary_mask(nnunet_dir / f"{case_id}_mask.nii.gz", fallback_prob=nnunet_prob)
    profound_mask = load_binary_mask(profound_dir / f"{case_id}_mask.nii.gz", fallback_prob=profound_prob)
    return {
        "nnunet_prob": nnunet_prob,
        "profound_prob": profound_prob,
        "nnunet_mask": nnunet_mask,
        "profound_mask": profound_mask,
    }


class FusionCaseDataset(Dataset):
    def __init__(self, split: str, case_ids: Sequence[str]):
        self.split = split
        self.case_ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_id = self.case_ids[index]
        bundle = load_base_bundle(self.split, case_id)
        gt_array, _ = read_nifti(profound_label_path(self.split, case_id))
        gt_array = (gt_array > 0).astype(np.float32)
        stacked = np.stack([bundle["nnunet_prob"], bundle["profound_prob"]], axis=0).astype(np.float32)
        return {
            "case_id": case_id,
            "inputs": torch.from_numpy(stacked),
            "target": torch.from_numpy(gt_array).unsqueeze(0),
        }


class LoMixFusionNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels // 2),
            nn.GELU(),
        )
        self.head = nn.Conv3d(base_channels // 2, 1, kernel_size=1)
        nn.init.constant_(self.head.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x + self.block1(x)
        x = self.block2(x)
        return self.head(x)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, logits.dim()))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    return (alpha * torch.pow(1.0 - p_t, gamma) * bce).mean()


@dataclass
class TrainResult:
    best_val_dice: float
    best_threshold: float


def train_lomix(train_ids: Sequence[str], val_ids: Sequence[str], args: argparse.Namespace, device: torch.device) -> TrainResult:
    train_ds = FusionCaseDataset("train", train_ids)
    val_ds = FusionCaseDataset("train", val_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = LoMixFusionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    pos_weight = torch.tensor([args.pos_weight], device=device, dtype=torch.float32)

    best_state = None
    best_val_dice = -1.0
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"lomix-train-ep{epoch}", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast("cuda"):
                    logits = model(inputs)
                    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                    loss = args.bce_weight * bce + args.dice_weight * soft_dice_loss(logits, target) + args.focal_weight * focal_loss(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                loss = args.bce_weight * bce + args.dice_weight * soft_dice_loss(logits, target) + args.focal_weight * focal_loss(logits, target)
                loss.backward()
                optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_probs: List[np.ndarray] = []
        val_targets: List[np.ndarray] = []
        with torch.inference_mode():
            for batch in val_loader:
                inputs = batch["inputs"].to(device, non_blocking=True)
                target = batch["target"].cpu().numpy()[0, 0]
                probs = torch.sigmoid(model(inputs)).cpu().numpy()[0, 0]
                val_probs.append(probs)
                val_targets.append(target)

        thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]
        threshold_scores = []
        for threshold in thresholds:
            dice_scores = [dice_from_binary(threshold_mask(prob, threshold), target) for prob, target in zip(val_probs, val_targets)]
            threshold_scores.append(float(np.mean(dice_scores)))
        local_best_idx = int(np.argmax(threshold_scores))
        local_best_dice = float(threshold_scores[local_best_idx])
        local_best_threshold = float(thresholds[local_best_idx])
        print(
            f"[lomix] epoch={epoch} train_loss={train_loss / max(len(train_loader), 1):.4f} val_dice={local_best_dice:.4f} threshold={local_best_threshold:.2f}"
        )
        if local_best_dice > best_val_dice:
            best_val_dice = local_best_dice
            best_threshold = local_best_threshold
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
                "best_threshold": best_threshold,
                "args": vars(args),
            }

    if best_state is None:
        raise RuntimeError("LoMix training did not produce a valid checkpoint")
    torch.save(best_state, LOMIX_CHECKPOINT_PATH)
    return TrainResult(best_val_dice=best_val_dice, best_threshold=best_threshold)


def load_lomix(device: torch.device) -> Tuple[LoMixFusionNet, float]:
    if not LOMIX_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing LoMix checkpoint: {LOMIX_CHECKPOINT_PATH}")
    payload = torch.load(LOMIX_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = LoMixFusionNet().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, float(payload.get("best_threshold", 0.5))


def collect_split_case_ids(split: str) -> List[str]:
    if split == "train":
        return list_case_ids(TRAIN_IMAGE_DIR)
    if split == "test":
        return list_case_ids(TEST_IMAGE_DIR)
    raise ValueError(split)


def fuse_boolean(nnunet_mask: np.ndarray, profound_mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "and":
        return np.logical_and(nnunet_mask > 0, profound_mask > 0).astype(np.uint8)
    if mode == "or":
        return np.logical_or(nnunet_mask > 0, profound_mask > 0).astype(np.uint8)
    raise ValueError(mode)


def fuse_average(nnunet_prob: np.ndarray, profound_prob: np.ndarray) -> np.ndarray:
    return 0.5 * nnunet_prob + 0.5 * profound_prob


def fuse_dst(nnunet_prob: np.ndarray, profound_prob: np.ndarray) -> np.ndarray:
    p1 = np.clip(nnunet_prob.astype(np.float32), 1e-6, 1.0 - 1e-6)
    p2 = np.clip(profound_prob.astype(np.float32), 1e-6, 1.0 - 1e-6)
    m1_fg, m1_bg = p1, 1.0 - p1
    m2_fg, m2_bg = p2, 1.0 - p2
    conflict = m1_fg * m2_bg + m1_bg * m2_fg
    denom = np.clip(1.0 - conflict, 1e-6, None)
    return np.clip((m1_fg * m2_fg) / denom, 0.0, 1.0)


def save_method_output(method: str, case_id: str, prob: np.ndarray, reference_image: sitk.Image, threshold: float) -> np.ndarray:
    method_dir = ENSEMBLE_PRED_DIR / "fused" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    mask = threshold_mask(prob, threshold)
    write_mask(mask, reference_image, method_dir / f"{case_id}.nii.gz")
    return mask


def evaluate_test_set(test_ids: Sequence[str], lomix_model: nn.Module, lomix_threshold: float, device: torch.device) -> Dict[str, object]:
    per_case: List[Dict[str, object]] = []
    method_names = ["nnunet", "profound", "and", "or", "average", "dst", "lomix"]
    method_dices: Dict[str, List[float]] = {name: [] for name in method_names}
    method_precision: Dict[str, List[float]] = {name: [] for name in method_names}
    method_recall: Dict[str, List[float]] = {name: [] for name in method_names}

    for case_id in tqdm(test_ids, desc="ensemble-eval"):
        image_path = TEST_IMAGE_DIR / f"{case_id}_0000.nii.gz"
        label_path = TEST_LABEL_DIR / f"{case_id}.nii.gz"
        reference_image = sitk.ReadImage(str(image_path))
        gt_array, _ = read_nifti(label_path)
        gt_array = (gt_array > 0).astype(np.uint8)

        bundle = load_base_bundle("test", case_id)
        nnunet_prob = bundle["nnunet_prob"]
        profound_prob = bundle["profound_prob"]
        nnunet_mask = bundle["nnunet_mask"]
        profound_mask = bundle["profound_mask"]

        outputs: Dict[str, np.ndarray] = {
            "nnunet": nnunet_mask,
            "profound": profound_mask,
            "and": fuse_boolean(nnunet_mask, profound_mask, "and"),
            "or": fuse_boolean(nnunet_mask, profound_mask, "or"),
        }

        avg_prob = fuse_average(nnunet_prob, profound_prob)
        dst_prob = fuse_dst(nnunet_prob, profound_prob)
        outputs["average"] = save_method_output("average", case_id, avg_prob, reference_image, 0.5)
        outputs["dst"] = save_method_output("dst", case_id, dst_prob, reference_image, 0.5)

        lomix_input = torch.from_numpy(np.stack([nnunet_prob, profound_prob], axis=0)).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            lomix_prob = torch.sigmoid(lomix_model(lomix_input)).squeeze(0).squeeze(0).cpu().numpy()
        outputs["lomix"] = save_method_output("lomix", case_id, lomix_prob, reference_image, lomix_threshold)

        case_metrics = {}
        for method_name, pred_mask in outputs.items():
            metrics = compute_metrics(pred_mask, gt_array)
            case_metrics[method_name] = metrics
            method_dices[method_name].append(metrics["dice"])
            method_precision[method_name].append(metrics["precision"])
            method_recall[method_name].append(metrics["recall"])

        per_case.append({"case_id": case_id, "metrics": case_metrics})

    leaderboard = []
    for method_name in method_names:
        leaderboard.append(
            {
                "method": method_name,
                "mean_dice": float(np.mean(method_dices[method_name])),
                "median_dice": float(np.median(method_dices[method_name])),
                "min_dice": float(np.min(method_dices[method_name])),
                "max_dice": float(np.max(method_dices[method_name])),
                "mean_precision": float(np.mean(method_precision[method_name])),
                "mean_recall": float(np.mean(method_recall[method_name])),
            }
        )
    leaderboard.sort(key=lambda row: row["mean_dice"], reverse=True)

    print("\n=== Phase 4 Ensemble Leaderboard ===")
    for row in leaderboard:
        print(
            f"{row['method']:<10} dice={row['mean_dice']:.4f} precision={row['mean_precision']:.4f} recall={row['mean_recall']:.4f}"
        )

    report = {
        "num_cases": len(test_ids),
        "lomix_threshold": lomix_threshold,
        "leaderboard": leaderboard,
        "per_case": per_case,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def prepare_caches(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs()
    print("[*] Preparing nnU-Net and ProFound Models...")
    nnunet_source = NnUNetProbabilitySource(NNUNET_MODEL_DIR, NNUNET_CHECKPOINT_NAME)
    profound_model = load_profound_model(device)

    train_ids = collect_split_case_ids("train")
    test_ids = collect_split_case_ids("test")
    
    if args.max_cases is not None:
        train_ids = train_ids[: args.max_cases]
        test_ids = test_ids[: args.max_cases]

    print("[*] Generating Soft Probabilities via nnU-Net (This will take time on first run)...")
    nnunet_source.generate_split(TRAIN_IMAGE_DIR, ENSEMBLE_CACHE_DIR / "train" / "nnunet", force=args.force)
    nnunet_source.generate_split(TEST_IMAGE_DIR, ENSEMBLE_CACHE_DIR / "test" / "nnunet", force=args.force)
    
    print("[*] Generating Soft Probabilities via ProFound...")
    generate_profound_split("train", train_ids, profound_model, device, force=args.force)
    generate_profound_split("test", test_ids, profound_model, device, force=args.force)


def train_phase(args: argparse.Namespace, device: torch.device) -> None:
    prepare_caches(args, device)
    train_ids = collect_split_case_ids("train")
    if args.max_cases is not None:
        train_ids = train_ids[: args.max_cases]
    train_ids, val_ids = split_train_val(train_ids, args.val_ratio, args.seed)
    train_lomix(train_ids, val_ids, args, device)


def evaluate_phase(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    if not (ENSEMBLE_CACHE_DIR / "test" / "nnunet").exists() or not (ENSEMBLE_CACHE_DIR / "test" / "profound").exists():
        prepare_caches(args, device)
    lomix_model, lomix_threshold = load_lomix(device)
    test_ids = collect_split_case_ids("test")
    if args.max_cases is not None:
        test_ids = test_ids[: args.max_cases]
    return evaluate_test_set(test_ids, lomix_model, lomix_threshold, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 ensemble fusion and comparison (mpMRI).")
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "prepare":
        prepare_caches(args, device)
        return
    if args.mode == "train":
        train_phase(args, device)
        return
    if args.mode == "evaluate":
        evaluate_phase(args, device)
        return

    prepare_caches(args, device)
    train_phase(args, device)
    evaluate_phase(args, device)


if __name__ == "__main__":
    main()