import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_PATH = SCRIPT_DIR / "01_train_hyper_lomix.py"
MPMRI_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_ROOT = MPMRI_DIR / "reports" / "hyper_lomix_mpmri" / "final_model"


def load_train_module():
    spec = importlib.util.spec_from_file_location("hyper_lomix_train", TRAIN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain final Hyper-LoMix model on train+val using locked best hyperparameters.")
    parser.add_argument("--search-dir", type=str, required=True)
    parser.add_argument("--final-run-name", type=str, default="final_model")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--epochs-policy", choices=["best_epoch", "configured_epochs"], default="best_epoch")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def choose_final_epochs(best_config: Dict[str, object], explicit_final_epochs: Optional[int], epochs_policy: str) -> int:
    if explicit_final_epochs is not None:
        return explicit_final_epochs
    if epochs_policy == "configured_epochs":
        return int(best_config.get("config", {}).get("epochs", best_config.get("epochs_run", 1)))
    return int(best_config.get("best_epoch", best_config.get("epochs_run", 1)))


def main() -> None:
    args = parse_args()
    train_module = load_train_module()

    search_dir = Path(args.search_dir)
    leaderboard_path = search_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard: {leaderboard_path}")

    leaderboard = json.loads(leaderboard_path.read_text())
    best_config = leaderboard.get("best_config")
    if not best_config:
        raise RuntimeError("Search leaderboard does not contain a best_config")

    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = DEFAULT_FINAL_ROOT / args.final_run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_train_ids: List[str] = train_module.list_case_ids(train_module.TRAIN_IMAGE_DIR)
    if args.max_cases is not None:
        all_train_ids = all_train_ids[: args.max_cases]

    train_module.set_seed(int(best_config.get("config", {}).get("seed", 42)))
    train_module.ensure_dirs(
        train_module.RunPaths(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            best_model_path=checkpoint_dir / "hyper_lomix_final.pt",
            metrics_path=run_dir / "final_training_summary.json",
            metadata_path=run_dir / "final_run_metadata.json",
            split_path=run_dir / "final_train_ids.json",
        )
    )
    train_module.validate_case_assets(all_train_ids)

    clinical_features = train_module.load_clinical_features(train_module.CLINICAL_CSV_PATH)
    dataset = train_module.HyperLoMixDataset("train", all_train_ids, clinical_features)
    batch_size = int(best_config.get("config", {}).get("batch_size", 1))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_channels = int(best_config.get("config", {}).get("base_channels", best_config.get("base_channels", 16)))
    model = train_module.ClinicalGuidedLoMixNet(base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_config.get("config", {}).get("lr", best_config.get("lr", 3e-4))),
        weight_decay=float(best_config.get("config", {}).get("weight_decay", best_config.get("weight_decay", 1e-4))),
    )
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    pos_weight = torch.tensor(
        [float(best_config.get("config", {}).get("pos_weight", best_config.get("pos_weight", 10.0)))],
        device=device,
        dtype=torch.float32,
    )

    bce_weight = float(best_config.get("config", {}).get("bce_weight", best_config.get("bce_weight", 1.0)))
    dice_weight = float(best_config.get("config", {}).get("dice_weight", best_config.get("dice_weight", 1.0)))
    focal_weight = float(best_config.get("config", {}).get("focal_weight", best_config.get("focal_weight", 0.5)))
    final_epochs = choose_final_epochs(best_config, args.final_epochs, args.epochs_policy)

    history: List[Dict[str, float]] = []
    for epoch in range(1, final_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"hyper-lomix-final-{epoch}", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast("cuda"):
                    logits, _ = model(inputs, clinical)
                    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                    loss = bce_weight * bce + dice_weight * train_module.soft_dice_loss(logits, target) + focal_weight * train_module.focal_loss(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(inputs, clinical)
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                loss = bce_weight * bce + dice_weight * train_module.soft_dice_loss(logits, target) + focal_weight * train_module.focal_loss(logits, target)
                loss.backward()
                optimizer.step()

            train_loss += float(loss.item())

        mean_train_loss = train_loss / max(len(train_loader), 1)
        history.append({"epoch": epoch, "train_loss": mean_train_loss})
        print(f"epoch={epoch:02d} train_loss={mean_train_loss:.4f}")

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "final_epochs": final_epochs,
        "frozen_threshold": float(best_config.get("best_threshold", 0.5)),
        "search_dir": str(search_dir),
        "best_config": best_config,
    }
    torch.save(checkpoint_payload, checkpoint_dir / "hyper_lomix_final.pt")

    train_ids_payload = {
        "source_search_dir": str(search_dir),
        "num_train_cases": len(all_train_ids),
        "train_ids": all_train_ids,
    }
    (run_dir / "final_train_ids.json").write_text(json.dumps(train_ids_payload, indent=2))

    summary = {
        "final_run_name": args.final_run_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_dir / "hyper_lomix_final.pt"),
        "search_dir": str(search_dir),
        "source_best_config": best_config,
        "frozen_threshold": float(best_config.get("best_threshold", 0.5)),
        "final_epochs": final_epochs,
        "epochs_policy": args.epochs_policy,
        "num_train_cases": len(all_train_ids),
        "history": history,
    }
    (run_dir / "final_training_summary.json").write_text(json.dumps(summary, indent=2))

    metadata = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "args": vars(args),
        "run_dir": str(run_dir),
        "source_leaderboard": str(leaderboard_path),
    }
    (run_dir / "final_run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()