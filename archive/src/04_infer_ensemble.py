import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
TEST_CACHE_DIR = WORKSPACE_DIR / "data" / "02_frozen_features" / "test"
OUT_DIR = WORKSPACE_DIR / "data" / "03_phase4_logits" / "test"
DEFAULT_CKPT = WORKSPACE_DIR / "checkpoints" / "phase3" / "best.pth"


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
        return self.fuse(torch.cat([profound_logits, sam_logits], dim=1))


class Phase3Model(nn.Module):
    def __init__(self, profound_in_chans: int = 768, target_size: int = 32) -> None:
        super().__init__()
        self.decoder = ProFoundDecoder3D(in_chans=profound_in_chans, target_size=target_size)
        self.lomix = LoMixHead3D()

    def forward(self, profound_feat: torch.Tensor, sam_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        profound_logits = self.decoder(profound_feat)
        lomix_logits = self.lomix(profound_logits, sam_logits)
        return {
            "profound_logits": profound_logits,
            "lomix_logits": lomix_logits,
        }


def stable_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def dst_fusion_logits(profound_logits: torch.Tensor, sam_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p1 = torch.sigmoid(profound_logits)
    p2 = torch.sigmoid(sam_logits)

    m1_l, m1_b = p1, 1.0 - p1
    m2_l, m2_b = p2, 1.0 - p2

    conflict = m1_l * m2_b + m1_b * m2_l
    denom = (1.0 - conflict).clamp_min(eps)

    m_l = (m1_l * m2_l) / denom
    return stable_logit(m_l, eps=eps)


class CachedTestDataset(Dataset):
    def __init__(self, cache_dir: Path, target_size: int, max_patients: int = 0):
        self.cache_dir = cache_dir
        ids = sorted({p.name[: -len("_profound.npy")] for p in cache_dir.glob("*_profound.npy")})
        if max_patients > 0:
            ids = ids[: max_patients]
        self.patient_ids = ids
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        profound = np.load(self.cache_dir / f"{pid}_profound.npy").astype(np.float32)
        sam_logits = np.load(self.cache_dir / f"{pid}_sam_logits.npy").astype(np.float32)

        profound_t = torch.from_numpy(profound)
        sam_t = torch.from_numpy(sam_logits)
        if sam_t.ndim == 3:
            sam_t = sam_t.unsqueeze(0)
        if list(sam_t.shape[-3:]) != [self.target_size, self.target_size, self.target_size]:
            sam_t = F.interpolate(
                sam_t.unsqueeze(0),
                size=(self.target_size, self.target_size, self.target_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        return {
            "patient_id": pid,
            "profound_feat": profound_t,
            "sam_logits": sam_t,
        }


def ensure_dirs(base_out: Path) -> Dict[str, Path]:
    out = {
        "simple_avg": base_out / "simple_avg",
        "lomix": base_out / "lomix",
        "dst": base_out / "dst",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 inference for SimpleAvg, LoMix, and DST logits.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Phase3Model(profound_in_chans=768, target_size=args.target_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = CachedTestDataset(TEST_CACHE_DIR, target_size=args.target_size, max_patients=args.max_patients)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    method_dirs = ensure_dirs(OUT_DIR)
    summary = {
        "checkpoint": str(ckpt_path),
        "target_size": args.target_size,
        "num_patients": len(ds),
        "methods": ["simple_avg", "lomix", "dst"],
        "outputs": {k: str(v) for k, v in method_dirs.items()},
    }

    with torch.no_grad():
        for batch in loader:
            pids: List[str] = batch["patient_id"]
            profound_feat = batch["profound_feat"].to(device, non_blocking=True)
            sam_logits = batch["sam_logits"].to(device, non_blocking=True)

            out = model(profound_feat, sam_logits)
            profound_logits = out["profound_logits"]
            lomix_logits = out["lomix_logits"]
            avg_logits = 0.5 * (profound_logits + sam_logits)
            dst_logits = dst_fusion_logits(profound_logits, sam_logits)

            for i, pid in enumerate(pids):
                np.save(method_dirs["simple_avg"] / f"{pid}_logits.npy", avg_logits[i].squeeze(0).cpu().numpy())
                np.save(method_dirs["lomix"] / f"{pid}_logits.npy", lomix_logits[i].squeeze(0).cpu().numpy())
                np.save(method_dirs["dst"] / f"{pid}_logits.npy", dst_logits[i].squeeze(0).cpu().numpy())

    with open(OUT_DIR / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Phase 4 inference complete for {len(ds)} patients.")
    print(f"Outputs written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
