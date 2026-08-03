import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_PATH = SCRIPT_DIR / "01_train_hyper_lomix.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("hyper_lomix_train", TRAIN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the final Hyper-LoMix model on the held-out test set.")
    parser.add_argument("--final-run-dir", type=str, required=True)
    return parser.parse_args()


def precision_recall_from_binary(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    pred_b = (pred > 0).astype(np.uint8)
    target_b = (target > 0).astype(np.uint8)
    tp = int(np.sum((pred_b == 1) & (target_b == 1)))
    fp = int(np.sum((pred_b == 1) & (target_b == 0)))
    fn = int(np.sum((pred_b == 0) & (target_b == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return float(precision), float(recall)


def summarize_metrics(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "median": float(np.median(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
    }


def load_test_case_arrays(train_module, case_id: str) -> Dict[str, np.ndarray]:
    nnunet_path = train_module.TEST_CACHE_DIR / "nnunet" / f"{case_id}_prob.npy"
    profound_path = train_module.TEST_CACHE_DIR / "profound" / f"{case_id}_prob.npy"
    label_path = train_module.TEST_LABEL_DIR / f"{case_id}.nii.gz"

    nnunet_prob = train_module.normalize_probability_volume(train_module.load_probability_file(nnunet_path))
    profound_prob = train_module.normalize_probability_volume(train_module.load_probability_file(profound_path))
    label_array, _ = train_module.read_nifti(label_path)
    label_mask = (label_array > 0).astype(np.float32)

    return {
        "nnunet_prob": train_module.resize_array_to_shape(nnunet_prob, train_module.TARGET_SHAPE),
        "profound_prob": train_module.resize_array_to_shape(profound_prob, train_module.TARGET_SHAPE),
        "target": train_module.resize_array_to_shape(label_mask, train_module.TARGET_SHAPE),
    }


def main() -> None:
    args = parse_args()
    train_module = load_train_module()

    final_run_dir = Path(args.final_run_dir)
    summary_path = final_run_dir / "final_training_summary.json"
    checkpoint_path = final_run_dir / "checkpoints" / "hyper_lomix_final.pt"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Final run summary or checkpoint is missing")

    final_summary = json.loads(summary_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    frozen_threshold = float(final_summary["frozen_threshold"])
    best_config = final_summary["source_best_config"]
    base_channels = int(best_config.get("config", {}).get("base_channels", best_config.get("base_channels", 16)))

    clinical_features = train_module.load_clinical_features(train_module.CLINICAL_CSV_PATH)
    test_ids = train_module.list_case_ids(train_module.TEST_IMAGE_DIR)
    test_dataset = train_module.HyperLoMixDataset("test", test_ids, clinical_features)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_module.ClinicalGuidedLoMixNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_case: List[Dict[str, object]] = []
    hyper_dices: List[float] = []
    nnunet_dices: List[float] = []
    profound_dices: List[float] = []
    hyper_precisions: List[float] = []
    hyper_recalls: List[float] = []

    with torch.inference_mode():
        for batch in test_loader:
            case_id = batch["case_id"][0]
            inputs = batch["inputs"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            target = batch["target"].cpu().numpy()[0, 0]
            logits, _ = model(inputs, clinical)
            hyper_prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            hyper_pred = train_module.threshold_mask(hyper_prob, frozen_threshold)
            hyper_dice = train_module.dice_from_binary(hyper_pred, target)
            hyper_precision, hyper_recall = precision_recall_from_binary(hyper_pred, target)

            case_arrays = load_test_case_arrays(train_module, case_id)
            nnunet_pred = train_module.threshold_mask(case_arrays["nnunet_prob"], frozen_threshold)
            profound_pred = train_module.threshold_mask(case_arrays["profound_prob"], frozen_threshold)
            nnunet_dice = train_module.dice_from_binary(nnunet_pred, target)
            profound_dice = train_module.dice_from_binary(profound_pred, target)

            hyper_dices.append(hyper_dice)
            nnunet_dices.append(nnunet_dice)
            profound_dices.append(profound_dice)
            hyper_precisions.append(hyper_precision)
            hyper_recalls.append(hyper_recall)

            per_case.append(
                {
                    "case_id": case_id,
                    "hyper_lomix_dice": hyper_dice,
                    "hyper_lomix_precision": hyper_precision,
                    "hyper_lomix_recall": hyper_recall,
                    "nnunet_dice": nnunet_dice,
                    "profound_dice": profound_dice,
                }
            )

    report = {
        "final_run_dir": str(final_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "frozen_threshold": frozen_threshold,
        "num_test_cases": len(test_ids),
        "hyper_lomix": {
            "dice": summarize_metrics(hyper_dices),
            "precision": summarize_metrics(hyper_precisions),
            "recall": summarize_metrics(hyper_recalls),
        },
        "nnunet": {
            "dice": summarize_metrics(nnunet_dices),
        },
        "profound": {
            "dice": summarize_metrics(profound_dices),
        },
        "per_case": per_case,
    }

    report_path = final_run_dir / "test_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()