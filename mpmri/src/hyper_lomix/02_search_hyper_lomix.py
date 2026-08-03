import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


WORKSPACE_DIR = Path(__file__).resolve().parents[3]
MPMRI_DIR = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).with_name("01_train_hyper_lomix.py")
DEFAULT_RUN_ROOT = MPMRI_DIR / "reports" / "hyper_lomix_mpmri" / "searches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Hyper-LoMix hyperparameter search on a fixed train/val split.")
    parser.add_argument("--search-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, nargs="+", default=[5])
    parser.add_argument("--base-channels", type=int, nargs="+", default=[16])
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--pos-weights", type=float, nargs="+", default=[10.0])
    parser.add_argument("--bce-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--dice-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--focal-weights", type=float, nargs="+", default=[0.5])
    parser.add_argument("--run-root", type=str, default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed trials in the search directory and only run missing ones.",
    )
    return parser.parse_args()


def build_trials(args: argparse.Namespace) -> List[Dict[str, float]]:
    trials: List[Dict[str, float]] = []
    grid: Iterable[tuple] = itertools.product(
        args.lrs,
        args.weight_decays,
        args.pos_weights,
        args.bce_weights,
        args.dice_weights,
        args.focal_weights,
        args.base_channels,
        args.patience,
    )
    for index, values in enumerate(grid, start=1):
        (
            lr,
            weight_decay,
            pos_weight,
            bce_weight,
            dice_weight,
            focal_weight,
            base_channels,
            patience,
        ) = values
        trials.append(
            {
                "trial_index": index,
                "lr": lr,
                "weight_decay": weight_decay,
                "pos_weight": pos_weight,
                "bce_weight": bce_weight,
                "dice_weight": dice_weight,
                "focal_weight": focal_weight,
                "base_channels": int(base_channels),
                "patience": int(patience),
            }
        )
    return trials


def run_trial(
    args: argparse.Namespace,
    search_dir: Path,
    split_path: Path,
    trial: Dict[str, float],
) -> Dict[str, object]:
    run_name = f"trial_{trial['trial_index']:03d}"
    trial_dir = search_dir / "trials" / run_name
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--lr",
        str(trial["lr"]),
        "--weight-decay",
        str(trial["weight_decay"]),
        "--pos-weight",
        str(trial["pos_weight"]),
        "--bce-weight",
        str(trial["bce_weight"]),
        "--dice-weight",
        str(trial["dice_weight"]),
        "--focal-weight",
        str(trial["focal_weight"]),
        "--base-channels",
        str(trial["base_channels"]),
        "--patience",
        str(trial["patience"]),
        "--run-name",
        run_name,
        "--output-dir",
        str(trial_dir),
        "--split-path",
        str(split_path),
    ]
    if args.max_cases is not None:
        command.extend(["--max-cases", str(args.max_cases)])

    subprocess.run(command, cwd=WORKSPACE_DIR, check=True)
    summary_path = trial_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    result = {
        **trial,
        "run_name": run_name,
        "trial_dir": str(trial_dir),
        "best_val_dice": summary["best_val_dice"],
        "best_val_loss": summary["best_val_loss"],
        "best_threshold": summary["best_threshold"],
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "num_train_cases": summary["num_train_cases"],
        "num_val_cases": summary["num_val_cases"],
        "nnunet_val_dice": summary["baselines_at_hyper_threshold"]["nnunet_val_dice"],
        "profound_val_dice": summary["baselines_at_hyper_threshold"]["profound_val_dice"],
    }
    return result


def result_from_existing_summary(search_dir: Path, trial: Dict[str, float]) -> Dict[str, object] | None:
    run_name = f"trial_{trial['trial_index']:03d}"
    trial_dir = search_dir / "trials" / run_name
    summary_path = trial_dir / "training_summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text())
    return {
        **trial,
        "run_name": run_name,
        "trial_dir": str(trial_dir),
        "best_val_dice": summary["best_val_dice"],
        "best_val_loss": summary["best_val_loss"],
        "best_threshold": summary["best_threshold"],
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "num_train_cases": summary["num_train_cases"],
        "num_val_cases": summary["num_val_cases"],
        "nnunet_val_dice": summary["baselines_at_hyper_threshold"]["nnunet_val_dice"],
        "profound_val_dice": summary["baselines_at_hyper_threshold"]["profound_val_dice"],
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    search_dir = Path(args.run_root) / args.search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    split_path = search_dir / "shared_train_val_split.json"

    trials = build_trials(args)
    leaderboard: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for trial in trials:
        if args.resume:
            existing = result_from_existing_summary(search_dir, trial)
            if existing is not None:
                leaderboard.append(existing)
                continue
        try:
            result = run_trial(args, search_dir, split_path, trial)
            leaderboard.append(result)
        except subprocess.CalledProcessError as exc:
            failures.append(
                {
                    **trial,
                    "returncode": exc.returncode,
                    "run_name": f"trial_{trial['trial_index']:03d}",
                }
            )

    leaderboard.sort(key=lambda row: (-float(row["best_val_dice"]), float(row["best_val_loss"])))

    summary = {
        "search_name": args.search_name,
        "search_dir": str(search_dir),
        "split_path": str(split_path),
        "num_trials": len(trials),
        "num_successful_trials": len(leaderboard),
        "num_failed_trials": len(failures),
        "ranking_rule": "best_val_dice desc, best_val_loss asc",
        "args": vars(args),
        "leaderboard": leaderboard,
        "failures": failures,
        "best_config": leaderboard[0] if leaderboard else None,
    }

    (search_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2))
    write_csv(search_dir / "leaderboard.csv", leaderboard)
    if leaderboard:
        (search_dir / "best_config.json").write_text(json.dumps(leaderboard[0], indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()