import argparse
import csv
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass
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

try:
    torch.serialization.add_safe_globals([
        np.dtype,
        np.bool_,
        np.int64,
        np.int32,
        np.float32,
        np.float64,
        np._core.multiarray.scalar,
        np.core.multiarray.scalar,
    ])
except Exception:
    pass

_original_torch_load = torch.load


def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _compat_torch_load


WORKSPACE_DIR = Path(__file__).resolve().parents[3]
MPMRI_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = MPMRI_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset501_ProstateMPMRI"
TRAIN_IMAGE_DIR = DATASET_DIR / "imagesTr"
TRAIN_LABEL_DIR = DATASET_DIR / "labelsTr"
TEST_IMAGE_DIR = DATASET_DIR / "imagesTs"
TEST_LABEL_DIR = DATASET_DIR / "labelsTs"

CLINICAL_CSV_PATH = WORKSPACE_DIR / "data" / "01_promis_raw" / "promis_mapped" / "lesion_ordered.csv"
CACHE_DIR = MPMRI_DIR / "reports" / "ensemble_cache_mpmri"
TRAIN_CACHE_DIR = CACHE_DIR / "train"
TEST_CACHE_DIR = CACHE_DIR / "test"

REPORT_DIR = MPMRI_DIR / "reports" / "hyper_lomix_mpmri"
CHECKPOINT_DIR = MPMRI_DIR / "checkpoints" / "hyper_lomix_mpmri"
SPLIT_PATH = REPORT_DIR / "train_val_split.json"

TARGET_SHAPE = (64, 160, 160)
EPS = 1e-6


@dataclass
class TrainConfig:
    seed: int = 42
    val_ratio: float = 0.15
    epochs: int = 10
    batch_size: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pos_weight: float = 10.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    max_cases: Optional[int] = None
    patience: int = 5
    base_channels: int = 16
    run_name: Optional[str] = None


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    best_model_path: Path
    metrics_path: Path
    metadata_path: Path
    split_path: Path


class ClinicalGuidedLoMixNet(nn.Module):
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
        self.clinical_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 4 * 4 * 4),
        )
        self.output_logit = nn.Conv3d(base_channels // 2 + 1, 1, kernel_size=1)
        nn.init.constant_(self.output_logit.bias, -2.5)

    def _build_gate(self, clinical_vector: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        batch_size = clinical_vector.shape[0]
        latent = self.clinical_mlp(clinical_vector)
        params = self.gate_head(latent).view(batch_size, 1, 4, 4, 4)
        gate = F.interpolate(params, size=target_shape, mode="trilinear", align_corners=False)
        return torch.sigmoid(gate)

    def forward(self, probs: torch.Tensor, clinical_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.stem(probs)
        spatial = spatial + self.block1(spatial)
        spatial = self.block2(spatial)
        gate = self._build_gate(clinical_vector, target_shape=(probs.shape[2], probs.shape[3], probs.shape[4]))
        p_nn = probs[:, 0:1]
        p_pf = probs[:, 1:2]
        fused_prob = gate * p_nn + (1.0 - gate) * p_pf
        merged = torch.cat([spatial, fused_prob], dim=1)
        logits = self.output_logit(merged)
        return logits, gate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_paths(run_name: Optional[str], output_dir: Optional[str], split_path: Optional[str]) -> RunPaths:
    if output_dir:
        run_dir = Path(output_dir)
    elif run_name:
        run_dir = REPORT_DIR / "runs" / run_name
    else:
        run_dir = REPORT_DIR

    checkpoint_dir = run_dir / "checkpoints"
    resolved_split_path = Path(split_path) if split_path else SPLIT_PATH
    return RunPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        best_model_path=checkpoint_dir / "hyper_lomix_best.pt",
        metrics_path=run_dir / "training_summary.json",
        metadata_path=run_dir / "run_metadata.json",
        split_path=resolved_split_path,
    )


def ensure_dirs(paths: RunPaths) -> None:
    for path in [paths.run_dir, paths.checkpoint_dir, paths.split_path.parent]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKSPACE_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


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


def resize_array_to_shape(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(array.shape) == target_shape:
        return array
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy()


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


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, logits.dim()))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    return (alpha * torch.pow(1.0 - p_t, gamma) * bce).mean()


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


def load_or_create_split(case_ids: Sequence[str], val_ratio: float, seed: int, split_path: Path) -> Tuple[List[str], List[str]]:
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        train_ids = [case_id for case_id in payload.get("train_ids", []) if case_id in case_ids]
        val_ids = [case_id for case_id in payload.get("val_ids", []) if case_id in case_ids]
        if train_ids and val_ids:
            return sorted(train_ids), sorted(val_ids)

    train_ids, val_ids = split_train_val(case_ids, val_ratio, seed)
    split_payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }
    split_path.write_text(json.dumps(split_payload, indent=2))
    return train_ids, val_ids


def validate_case_assets(case_ids: Sequence[str]) -> None:
    missing_assets: List[str] = []
    for case_id in case_ids:
        required_paths = [
            TRAIN_CACHE_DIR / "nnunet" / f"{case_id}_prob.npy",
            TRAIN_CACHE_DIR / "profound" / f"{case_id}_prob.npy",
            TRAIN_LABEL_DIR / f"{case_id}.nii.gz",
        ]
        for required_path in required_paths:
            if not required_path.exists():
                missing_assets.append(str(required_path))

    if missing_assets:
        preview = "\n".join(missing_assets[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_assets)} required training assets. First missing paths:\n{preview}"
        )


def load_clinical_features(path: Path) -> Dict[str, Tuple[float, float]]:
    features: Dict[str, Tuple[float, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row.get("patient", "").strip()
            if not patient_id:
                continue
            max_isup_vals = []
            max_pirads_vals = []
            for idx in range(1, 4):
                isup_val = row.get(f"max_isup_{idx}", "")
                pirads_val = row.get(f"pirads_{idx}", "")
                if isup_val not in {"", "nan", "NaN", None}:
                    try:
                        max_isup_vals.append(float(isup_val))
                    except ValueError:
                        pass
                if pirads_val not in {"", "nan", "NaN", None}:
                    try:
                        max_pirads_vals.append(float(pirads_val))
                    except ValueError:
                        pass
            max_isup = max(max_isup_vals) if max_isup_vals else 0.0
            max_pirads = max(max_pirads_vals) if max_pirads_vals else 0.0
            features[patient_id] = (max_isup / 5.0, max_pirads / 5.0)
    return features


class HyperLoMixDataset(Dataset):
    def __init__(self, split: str, case_ids: Sequence[str], clinical_features: Dict[str, Tuple[float, float]]):
        self.split = split
        self.case_ids = list(case_ids)
        self.clinical_features = clinical_features

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_id = self.case_ids[index]
        cache_dir = TRAIN_CACHE_DIR if self.split == "train" else TEST_CACHE_DIR
        nnunet_dir = cache_dir / "nnunet"
        profound_dir = cache_dir / "profound"

        nnunet_prob = load_probability_file(nnunet_dir / f"{case_id}_prob.npy")
        profound_prob = load_probability_file(profound_dir / f"{case_id}_prob.npy")

        label_path = TRAIN_LABEL_DIR / f"{case_id}.nii.gz" if self.split == "train" else TEST_LABEL_DIR / f"{case_id}.nii.gz"
        if label_path.exists():
            label_array, _ = read_nifti(label_path)
            label_mask = (label_array > 0).astype(np.float32)
        else:
            mask_path = nnunet_dir / f"{case_id}_mask.nii.gz"
            if mask_path.exists():
                label_array, _ = read_nifti(mask_path)
                label_mask = (label_array > 0).astype(np.float32)
            else:
                label_mask = (np.zeros_like(nnunet_prob, dtype=np.uint8) > 0).astype(np.float32)

        nnunet_prob = resize_array_to_shape(nnunet_prob, TARGET_SHAPE)
        profound_prob = resize_array_to_shape(profound_prob, TARGET_SHAPE)
        label_mask = resize_array_to_shape(label_mask, TARGET_SHAPE)

        stacked = np.stack([nnunet_prob, profound_prob], axis=0).astype(np.float32)
        clinical_vector = np.asarray(self.clinical_features.get(case_id, (0.0, 0.0)), dtype=np.float32)

        sample = {
            "case_id": case_id,
            "inputs": torch.from_numpy(stacked),
            "clinical": torch.from_numpy(clinical_vector),
            "target": torch.from_numpy(label_mask).unsqueeze(0),
        }
        return sample


def load_case_arrays(case_id: str) -> Dict[str, np.ndarray]:
    nnunet_path = TRAIN_CACHE_DIR / "nnunet" / f"{case_id}_prob.npy"
    profound_path = TRAIN_CACHE_DIR / "profound" / f"{case_id}_prob.npy"
    label_path = TRAIN_LABEL_DIR / f"{case_id}.nii.gz"

    nnunet_prob = normalize_probability_volume(load_probability_file(nnunet_path))
    profound_prob = normalize_probability_volume(load_probability_file(profound_path))
    label_array, _ = read_nifti(label_path)
    label_mask = (label_array > 0).astype(np.float32)

    return {
        "nnunet_prob": resize_array_to_shape(nnunet_prob, TARGET_SHAPE),
        "profound_prob": resize_array_to_shape(profound_prob, TARGET_SHAPE),
        "target": resize_array_to_shape(label_mask, TARGET_SHAPE),
    }


def sweep_best_threshold(probs_list: Sequence[np.ndarray], targets_list: Sequence[np.ndarray]) -> Tuple[float, float]:
    thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]
    best_threshold = 0.5
    best_dice = -1.0
    for threshold in thresholds:
        dice_scores = [
            dice_from_binary(threshold_mask(prob, threshold), target)
            for prob, target in zip(probs_list, targets_list)
        ]
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = float(threshold)
    return best_threshold, best_dice


def train_model(
    config: TrainConfig,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    clinical_features: Dict[str, Tuple[float, float]],
    paths: RunPaths,
) -> Dict[str, object]:
    ensure_dirs(paths)
    train_ds = HyperLoMixDataset("train", train_ids, clinical_features)
    val_ds = HyperLoMixDataset("train", val_ids, clinical_features)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalGuidedLoMixNet(base_channels=config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    pos_weight = torch.tensor([config.pos_weight], device=device, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"hyper-lomix-epoch-{epoch}", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast("cuda"):
                    logits, _ = model(inputs, clinical)
                    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                    loss = config.bce_weight * bce + config.dice_weight * soft_dice_loss(logits, target) + config.focal_weight * focal_loss(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(inputs, clinical)
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                loss = config.bce_weight * bce + config.dice_weight * soft_dice_loss(logits, target) + config.focal_weight * focal_loss(logits, target)
                loss.backward()
                optimizer.step()

            train_loss += float(loss.item())

        model.eval()
        val_loss_total = 0.0
        val_probs: List[np.ndarray] = []
        val_targets: List[np.ndarray] = []
        with torch.inference_mode():
            for batch in val_loader:
                inputs = batch["inputs"].to(device, non_blocking=True)
                clinical = batch["clinical"].to(device, non_blocking=True)
                target_tensor = batch["target"].to(device, non_blocking=True)
                logits, _ = model(inputs, clinical)
                bce = F.binary_cross_entropy_with_logits(logits, target_tensor, pos_weight=pos_weight)
                loss = config.bce_weight * bce + config.dice_weight * soft_dice_loss(logits, target_tensor) + config.focal_weight * focal_loss(logits, target_tensor)
                val_loss_total += float(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                target = target_tensor.cpu().numpy()[0, 0]
                val_probs.append(probs)
                val_targets.append(target)

        val_loss = val_loss_total / max(len(val_loader), 1)
        history.append({"epoch": epoch, "train_loss": train_loss / max(len(train_loader), 1), "val_loss": val_loss})
        print(f"epoch={epoch:02d} train_loss={train_loss / max(len(train_loader), 1):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    torch.save(best_state, paths.best_model_path)

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    val_hyper_probs: List[np.ndarray] = []
    val_targets: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in val_loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            target = batch["target"].cpu().numpy()[0, 0]
            logits, _ = model(inputs, clinical)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            val_hyper_probs.append(probs)
            val_targets.append(target)

    best_threshold, hyper_val_dice = sweep_best_threshold(val_hyper_probs, val_targets)

    val_nnunet_probs: List[np.ndarray] = []
    val_profound_probs: List[np.ndarray] = []
    for case_id in val_ids:
        case_arrays = load_case_arrays(case_id)
        val_nnunet_probs.append(case_arrays["nnunet_prob"])
        val_profound_probs.append(case_arrays["profound_prob"])

    nnunet_val_dice = float(np.mean([
        dice_from_binary(threshold_mask(prob, best_threshold), target)
        for prob, target in zip(val_nnunet_probs, val_targets)
    ]))
    profound_val_dice = float(np.mean([
        dice_from_binary(threshold_mask(prob, best_threshold), target)
        for prob, target in zip(val_profound_probs, val_targets)
    ]))

    summary = {
        "model": "ClinicalGuidedLoMixNet",
        "split_path": str(paths.split_path),
        "run_name": config.run_name,
        "config": asdict(config),
        "paths": {
            "run_dir": str(paths.run_dir),
            "checkpoint_dir": str(paths.checkpoint_dir),
            "best_model_path": str(paths.best_model_path),
            "metrics_path": str(paths.metrics_path),
            "metadata_path": str(paths.metadata_path),
        },
        "best_val_loss": best_val_loss,
        "best_val_dice": hyper_val_dice,
        "best_threshold": best_threshold,
        "best_epoch": best_state["epoch"],
        "baselines_at_hyper_threshold": {
            "nnunet_val_dice": nnunet_val_dice,
            "profound_val_dice": profound_val_dice,
        },
        "num_train_cases": len(train_ids),
        "num_val_cases": len(val_ids),
        "epochs_run": len(history),
        "history": history,
    }
    write_json(paths.metrics_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clinical-guided LoMix-style fusion model on mpMRI cached probabilities.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = resolve_run_paths(args.run_name, args.output_dir, args.split_path)
    ensure_dirs(paths)
    clinical_features = load_clinical_features(CLINICAL_CSV_PATH)
    train_ids = list_case_ids(TRAIN_IMAGE_DIR)
    if args.max_cases is not None:
        train_ids = train_ids[: args.max_cases]
    train_ids, val_ids = load_or_create_split(train_ids, args.val_ratio, args.seed, paths.split_path)
    validate_case_assets([*train_ids, *val_ids])
    config = TrainConfig(
        seed=args.seed,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        max_cases=args.max_cases,
        patience=args.patience,
        base_channels=args.base_channels,
        run_name=args.run_name,
    )
    metadata = {
        "run_name": args.run_name,
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "git_commit": get_git_commit(),
        "paths": {
            "workspace_dir": str(WORKSPACE_DIR),
            "dataset_dir": str(DATASET_DIR),
            "cache_dir": str(CACHE_DIR),
            "run_dir": str(paths.run_dir),
            "split_path": str(paths.split_path),
        },
        "args": vars(args),
        "num_available_cases": len(list_case_ids(TRAIN_IMAGE_DIR)),
        "num_selected_cases": len(train_ids) + len(val_ids),
        "num_train_cases": len(train_ids),
        "num_val_cases": len(val_ids),
    }
    write_json(paths.metadata_path, metadata)
    summary = train_model(config, train_ids, val_ids, clinical_features, paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
