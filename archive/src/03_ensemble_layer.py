import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
TRAIN_CACHE_DIR = WORKSPACE_DIR / "data" / "02_frozen_features" / "train"
TRAIN_LABEL_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "labelsTr"

CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints" / "phase3"
REPORT_DIR = WORKSPACE_DIR / "reports"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def log_status(message: str) -> None:
	print(message, flush=True)


def split_ids(ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
	ids = sorted(ids)
	rng = random.Random(seed)
	rng.shuffle(ids)

	val_count = max(1, int(len(ids) * val_ratio)) if len(ids) > 1 else 0
	val_ids = ids[:val_count]
	train_ids = ids[val_count:]
	if not train_ids:
		train_ids = val_ids
	return train_ids, val_ids


def collect_train_ids(cache_dir: Path, label_dir: Path) -> List[str]:
	cache_ids = {p.name[: -len("_profound.npy")] for p in cache_dir.glob("*_profound.npy")}
	label_ids = {p.name[: -len(".nii.gz")] for p in label_dir.glob("*.nii.gz")}
	return sorted(cache_ids & label_ids)


def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	target = target.float()
	dims = (1, 2, 3, 4)
	intersection = (probs * target).sum(dims)
	denom = probs.sum(dims) + target.sum(dims)
	dice = (2.0 * intersection + eps) / (denom + eps)
	return 1.0 - dice.mean()


def hard_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
	pred = (torch.sigmoid(logits) > threshold).float()
	target = target.float()
	intersection = (pred * target).sum(dim=(1, 2, 3, 4))
	denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
	dice = (2.0 * intersection + eps) / (denom + eps)
	return float(dice.mean().item())


def resize_volume(volume: torch.Tensor, target_size: int, mode: str) -> torch.Tensor:
	# Input expected as [1, D, H, W] and returned as [1, target, target, target]
	volume_5d = volume.unsqueeze(0)
	resized = F.interpolate(
		volume_5d,
		size=(target_size, target_size, target_size),
		mode=mode,
		align_corners=False if mode in {"trilinear", "linear", "bilinear", "bicubic"} else None,
	)
	return resized.squeeze(0)


class CachedFusionDataset(Dataset):
	def __init__(
		self,
		cache_dir: Path,
		label_dir: Path,
		patient_ids: List[str],
		target_size: int,
	) -> None:
		self.cache_dir = cache_dir
		self.label_dir = label_dir
		self.patient_ids = patient_ids
		self.target_size = target_size

	def __len__(self) -> int:
		return len(self.patient_ids)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		pid = self.patient_ids[idx]

		profound_path = self.cache_dir / f"{pid}_profound.npy"
		sam_logits_path = self.cache_dir / f"{pid}_sam_logits.npy"
		label_path = self.label_dir / f"{pid}.nii.gz"

		profound = np.load(profound_path).astype(np.float32)
		sam_logits = np.load(sam_logits_path).astype(np.float32)

		if profound.ndim != 4:
			raise ValueError(f"Expected ProFound tensor rank 4 for {pid}, got shape {profound.shape}")
		profound_t = torch.from_numpy(profound)

		if sam_logits.ndim == 3:
			sam_logits_t = torch.from_numpy(sam_logits).unsqueeze(0)
		elif sam_logits.ndim == 4 and sam_logits.shape[0] == 1:
			sam_logits_t = torch.from_numpy(sam_logits)
		else:
			raise ValueError(f"Unexpected SAM logits shape for {pid}: {sam_logits.shape}")

		if list(sam_logits_t.shape[-3:]) != [self.target_size, self.target_size, self.target_size]:
			sam_logits_t = resize_volume(sam_logits_t, self.target_size, mode="trilinear")

		lbl_img = sitk.ReadImage(str(label_path))
		lbl_arr = sitk.GetArrayFromImage(lbl_img).astype(np.float32)
		lbl_arr = (lbl_arr > 0).astype(np.float32)
		gt_t = torch.from_numpy(lbl_arr).unsqueeze(0)
		if list(gt_t.shape[-3:]) != [self.target_size, self.target_size, self.target_size]:
			gt_t = resize_volume(gt_t, self.target_size, mode="nearest")

		return {
			"patient_id": pid,
			"profound_feat": profound_t,
			"sam_logits": sam_logits_t,
			"gt_mask": gt_t,
		}


class ConvBlock3D(nn.Module):
	def __init__(self, in_ch: int, out_ch: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm3d(out_ch, affine=True),
			nn.GELU(),
			nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm3d(out_ch, affine=True),
			nn.GELU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class ProFoundDecoder3D(nn.Module):
	def __init__(self, in_chans: int = 768, target_size: int = 32) -> None:
		super().__init__()
		self.target_size = target_size
		self.proj = nn.Conv3d(in_chans, 256, kernel_size=1)
		self.block1 = ConvBlock3D(256, 128)
		self.block2 = ConvBlock3D(128, 64)
		self.out_head = nn.Conv3d(64, 1, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.proj(x)
		x = F.interpolate(
			x,
			size=(self.target_size, self.target_size, self.target_size),
			mode="trilinear",
			align_corners=False,
		)
		x = self.block1(x)
		x = self.block2(x)
		return self.out_head(x)


class LoMixHead3D(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.fuse = nn.Conv3d(2, 1, kernel_size=1)

	def forward(self, profound_logits: torch.Tensor, sam_logits: torch.Tensor) -> torch.Tensor:
		x = torch.cat([profound_logits, sam_logits], dim=1)
		return self.fuse(x)


class Phase3Model(nn.Module):
	def __init__(self, profound_in_chans: int = 768, target_size: int = 32) -> None:
		super().__init__()
		self.decoder = ProFoundDecoder3D(in_chans=profound_in_chans, target_size=target_size)
		self.lomix = LoMixHead3D()

	def forward(self, profound_feat: torch.Tensor, sam_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
		profound_logits = self.decoder(profound_feat)
		fused_logits = self.lomix(profound_logits, sam_logits)
		return {"profound_logits": profound_logits, "fused_logits": fused_logits}


def build_dataloaders(
	target_size: int,
	batch_size: int,
	num_workers: int,
	val_ratio: float,
	seed: int,
	overfit_n: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
	ids = collect_train_ids(TRAIN_CACHE_DIR, TRAIN_LABEL_DIR)
	if not ids:
		raise RuntimeError("No train IDs found. Ensure Phase 2 cache and labelsTr are available.")

	if overfit_n > 0:
		ids = ids[: min(overfit_n, len(ids))]
		train_ids = ids
		val_ids = ids
	else:
		train_ids, val_ids = split_ids(ids, val_ratio=val_ratio, seed=seed)

	train_ds = CachedFusionDataset(TRAIN_CACHE_DIR, TRAIN_LABEL_DIR, train_ids, target_size=target_size)
	val_ds = CachedFusionDataset(TRAIN_CACHE_DIR, TRAIN_LABEL_DIR, val_ids, target_size=target_size)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	stats = {
		"all_ids": len(ids),
		"train_ids": len(train_ids),
		"val_ids": len(val_ids),
	}
	return train_loader, val_loader, stats


def run_epoch_train(
	model: Phase3Model,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	scaler: amp.GradScaler,
	device: torch.device,
	bce_weight: float,
	dice_weight: float,
	aux_weight: float,
	use_amp: bool,
) -> Dict[str, float]:
	model.train()
	total_loss = 0.0
	total_batches = 0

	bce_fn = nn.BCEWithLogitsLoss()

	for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
		profound_feat = batch["profound_feat"].to(device, non_blocking=True)
		sam_logits = batch["sam_logits"].to(device, non_blocking=True)
		gt_mask = batch["gt_mask"].to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		with amp.autocast(device_type=device.type, enabled=use_amp):
			out = model(profound_feat, sam_logits)
			fused_logits = out["fused_logits"]
			profound_logits = out["profound_logits"]

			bce = bce_fn(fused_logits, gt_mask)
			dice = soft_dice_loss_from_logits(fused_logits, gt_mask)
			aux = bce_fn(profound_logits, gt_mask)
			loss = bce_weight * bce + dice_weight * dice + aux_weight * aux

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		total_loss += float(loss.item())
		total_batches += 1
		if batch_idx == 1:
			log_status("[train] first batch processed")

	mean_loss = total_loss / max(total_batches, 1)
	return {"train_loss": mean_loss}


@torch.no_grad()
def run_epoch_val(
	model: Phase3Model,
	loader: DataLoader,
	device: torch.device,
	bce_weight: float,
	dice_weight: float,
	aux_weight: float,
) -> Dict[str, float]:
	model.eval()
	bce_fn = nn.BCEWithLogitsLoss()

	total_loss = 0.0
	total_hard_dice = 0.0
	total_batches = 0

	for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False), start=1):
		profound_feat = batch["profound_feat"].to(device, non_blocking=True)
		sam_logits = batch["sam_logits"].to(device, non_blocking=True)
		gt_mask = batch["gt_mask"].to(device, non_blocking=True)

		out = model(profound_feat, sam_logits)
		fused_logits = out["fused_logits"]
		profound_logits = out["profound_logits"]

		bce = bce_fn(fused_logits, gt_mask)
		dice = soft_dice_loss_from_logits(fused_logits, gt_mask)
		aux = bce_fn(profound_logits, gt_mask)
		loss = bce_weight * bce + dice_weight * dice + aux_weight * aux

		hard_dice = hard_dice_from_logits(fused_logits, gt_mask)

		total_loss += float(loss.item())
		total_hard_dice += hard_dice
		total_batches += 1
		if batch_idx == 1:
			log_status("[val] first batch processed")

	return {
		"val_loss": total_loss / max(total_batches, 1),
		"val_hard_dice": total_hard_dice / max(total_batches, 1),
	}


def save_checkpoint(
	path: Path,
	model: Phase3Model,
	optimizer: torch.optim.Optimizer,
	scaler: amp.GradScaler,
	epoch: int,
	best_val_dice: float,
	args: argparse.Namespace,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"epoch": epoch,
			"best_val_dice": best_val_dice,
			"model_state": model.state_dict(),
			"optimizer_state": optimizer.state_dict(),
			"scaler_state": scaler.state_dict(),
			"args": vars(args),
		},
		path,
	)


def load_checkpoint(
	resume_path: Path,
	model: Phase3Model,
	optimizer: torch.optim.Optimizer,
	scaler: amp.GradScaler,
	device: torch.device,
) -> Tuple[int, float]:
	ckpt = torch.load(resume_path, map_location=device)
	model.load_state_dict(ckpt["model_state"])
	optimizer.load_state_dict(ckpt["optimizer_state"])
	scaler.load_state_dict(ckpt["scaler_state"])
	return int(ckpt.get("epoch", 0)), float(ckpt.get("best_val_dice", -math.inf))


def main() -> None:
	log_status("Phase 3: starting script")
	parser = argparse.ArgumentParser(description="Phase 3: ProFound decoder + LoMix training.")
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--target-size", type=int, default=32)
	parser.add_argument("--overfit-n", type=int, default=0)
	parser.add_argument("--resume", type=str, default="")
	parser.add_argument("--bce-weight", type=float, default=1.0)
	parser.add_argument("--dice-weight", type=float, default=1.0)
	parser.add_argument("--aux-weight", type=float, default=0.2)
	args = parser.parse_args()

	log_status(f"Phase 3: arguments parsed -> epochs={args.epochs}, batch_size={args.batch_size}, target_size={args.target_size}, overfit_n={args.overfit_n}")
	set_seed(args.seed)
	log_status(f"Phase 3: seed set to {args.seed}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = device.type == "cuda"
	log_status(f"Phase 3: device ready -> {device}, amp={'on' if use_amp else 'off'}")

	log_status("Phase 3: building dataloaders")
	train_loader, val_loader, split_stats = build_dataloaders(
		target_size=args.target_size,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		val_ratio=args.val_ratio,
		seed=args.seed,
		overfit_n=args.overfit_n,
	)
	log_status(
		f"Phase 3: dataloaders ready -> all_ids={split_stats['all_ids']}, train_ids={split_stats['train_ids']}, val_ids={split_stats['val_ids']}"
	)

	log_status("Phase 3: building model and optimizer")
	model = Phase3Model(profound_in_chans=768, target_size=args.target_size).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scaler = amp.GradScaler(device=device.type, enabled=use_amp)
	log_status("Phase 3: model ready")

	start_epoch = 0
	best_val_dice = -math.inf
	if args.resume:
		log_status(f"Phase 3: resuming from {args.resume}")
		resume_path = Path(args.resume)
		if not resume_path.exists():
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
		start_epoch, best_val_dice = load_checkpoint(resume_path, model, optimizer, scaler, device)
		start_epoch += 1

	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	REPORT_DIR.mkdir(parents=True, exist_ok=True)
	train_log_path = REPORT_DIR / "phase3_train_log.jsonl"
	val_summary_path = REPORT_DIR / "phase3_val_metrics.json"

	if start_epoch == 0 and train_log_path.exists() and not args.resume:
		train_log_path.unlink()

	shape_snapshot_written = False
	history = []
	log_status("Phase 3: entering training loop")

	for epoch in range(start_epoch, args.epochs):
		log_status(f"Phase 3: epoch {epoch + 1}/{args.epochs} starting")
		train_metrics = run_epoch_train(
			model,
			train_loader,
			optimizer,
			scaler,
			device,
			bce_weight=args.bce_weight,
			dice_weight=args.dice_weight,
			aux_weight=args.aux_weight,
			use_amp=use_amp,
		)
		val_metrics = run_epoch_val(
			model,
			val_loader,
			device,
			bce_weight=args.bce_weight,
			dice_weight=args.dice_weight,
			aux_weight=args.aux_weight,
		)

		metrics = {
			"epoch": epoch,
			**split_stats,
			**train_metrics,
			**val_metrics,
			"best_val_hard_dice_so_far": max(best_val_dice, val_metrics["val_hard_dice"]),
		}
		history.append(metrics)

		with open(train_log_path, "a", encoding="utf-8") as f:
			f.write(json.dumps(metrics) + "\n")

		if not shape_snapshot_written:
			first_batch = next(iter(train_loader))
			snapshot = {
				"profound_feat": list(first_batch["profound_feat"].shape),
				"sam_logits": list(first_batch["sam_logits"].shape),
				"gt_mask": list(first_batch["gt_mask"].shape),
			}
			with open(REPORT_DIR / "phase3_shape_snapshot.json", "w", encoding="utf-8") as f:
				json.dump(snapshot, f, indent=2)
			shape_snapshot_written = True

		save_checkpoint(
			CHECKPOINT_DIR / "last.pth",
			model,
			optimizer,
			scaler,
			epoch,
			best_val_dice,
			args,
		)

		if val_metrics["val_hard_dice"] > best_val_dice:
			best_val_dice = val_metrics["val_hard_dice"]
			save_checkpoint(
				CHECKPOINT_DIR / "best.pth",
				model,
				optimizer,
				scaler,
				epoch,
				best_val_dice,
				args,
			)

		print(
			f"[Epoch {epoch}] train_loss={train_metrics['train_loss']:.5f} "
			f"val_loss={val_metrics['val_loss']:.5f} "
			f"val_hard_dice={val_metrics['val_hard_dice']:.5f} "
			f"best={best_val_dice:.5f}"
		)

	with open(val_summary_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"best_val_hard_dice": best_val_dice,
				"history": history,
				"split_stats": split_stats,
				"target_size": args.target_size,
			},
			f,
			indent=2,
		)

	log_status("Training complete.")
	log_status(f"Best checkpoint: {CHECKPOINT_DIR / 'best.pth'}")
	log_status(f"Last checkpoint: {CHECKPOINT_DIR / 'last.pth'}")
	log_status(f"Train log: {train_log_path}")
	log_status(f"Validation summary: {val_summary_path}")


if __name__ == "__main__":
	main()

