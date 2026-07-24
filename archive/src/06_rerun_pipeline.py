import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = WORKSPACE_DIR / "reports"
PHASE4_DIR = WORKSPACE_DIR / "data" / "03_phase4_logits" / "test"
PHASE5_DIR = WORKSPACE_DIR / "data" / "04_phase5_native_predictions"
PHASE3_DIR = WORKSPACE_DIR / "checkpoints" / "phase3"
PHASE3_LOGS = [
	REPORT_DIR / "phase3_train_log.jsonl",
	REPORT_DIR / "phase3_val_metrics.json",
	REPORT_DIR / "phase3_shape_snapshot.json",
	REPORT_DIR / "phase3_console.log",
	REPORT_DIR / "rerun_phase3.log",
	REPORT_DIR / "rerun_phase4.log",
	REPORT_DIR / "rerun_phase5.log",
]
RERUN_STATUS_PATH = REPORT_DIR / "rerun_pipeline_status.json"


def log(message: str) -> None:
	print(message, flush=True)


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def write_status(status: dict) -> None:
	REPORT_DIR.mkdir(parents=True, exist_ok=True)
	with open(RERUN_STATUS_PATH, "w", encoding="utf-8") as f:
		json.dump(status, f, indent=2)


def remove_path(path: Path) -> None:
	if path.is_dir():
		shutil.rmtree(path)
	elif path.exists():
		path.unlink()


def clean_outputs() -> None:
	for path in PHASE3_LOGS:
		if path.exists():
			log(f"Removing {path}")
			remove_path(path)
	if PHASE3_DIR.exists():
		for ckpt_name in ["best.pth", "last.pth"]:
			ckpt_path = PHASE3_DIR / ckpt_name
			if ckpt_path.exists():
				log(f"Removing {ckpt_path}")
				remove_path(ckpt_path)
	for path in [PHASE4_DIR, PHASE5_DIR]:
		if path.exists():
			log(f"Cleaning {path}")
			remove_path(path)


def stream_command(command: List[str], log_path: Path) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log(f"Running: {' '.join(command)}")
	with open(log_path, "w", encoding="utf-8") as log_file:
		process = subprocess.Popen(
			command,
			cwd=str(WORKSPACE_DIR),
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
		)
		assert process.stdout is not None
		for line in process.stdout:
			sys.stdout.write(line)
			sys.stdout.flush()
			log_file.write(line)
			log_file.flush()
		exit_code = process.wait()
		if exit_code != 0:
			raise subprocess.CalledProcessError(exit_code, command)


def build_phase3_command(args: argparse.Namespace) -> List[str]:
	command = [
		sys.executable,
		"src/03_ensemble_layer.py",
		"--epochs",
		str(args.epochs),
		"--batch-size",
		str(args.batch_size),
		"--num-workers",
		str(args.num_workers),
		"--lr",
		str(args.lr),
		"--weight-decay",
		str(args.weight_decay),
		"--val-ratio",
		str(args.val_ratio),
		"--seed",
		str(args.seed),
		"--target-size",
		str(args.target_size),
		"--bce-weight",
		str(args.bce_weight),
		"--dice-weight",
		str(args.dice_weight),
		"--aux-weight",
		str(args.aux_weight),
	]
	if args.overfit_n > 0:
		command.extend(["--overfit-n", str(args.overfit_n)])
	if args.resume:
		command.extend(["--resume", args.resume])
	return command


def build_phase4_command(args: argparse.Namespace) -> List[str]:
	command = [
		sys.executable,
		"src/04_infer_ensemble.py",
		"--checkpoint",
		str(args.checkpoint),
		"--target-size",
		str(args.target_size),
		"--batch-size",
		str(args.phase4_batch_size),
		"--num-workers",
		str(args.phase4_num_workers),
	]
	if args.phase4_max_patients > 0:
		command.extend(["--max-patients", str(args.phase4_max_patients)])
	return command


def build_phase5_command(args: argparse.Namespace) -> List[str]:
	command = [sys.executable, "src/05_reverse_align_eval.py"]
	if args.phase5_max_patients > 0:
		command.extend(["--max-patients", str(args.phase5_max_patients)])
	return command


def main() -> None:
	parser = argparse.ArgumentParser(description="Rerun Phase 3, Phase 4, and Phase 5 in sequence.")
	parser.add_argument("--clean", action="store_true", help="Remove existing Phase 4 and Phase 5 outputs before rerunning.")
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
	parser.add_argument("--checkpoint", type=str, default=str(WORKSPACE_DIR / "checkpoints" / "phase3" / "best.pth"))
	parser.add_argument("--phase4-batch-size", type=int, default=1)
	parser.add_argument("--phase4-num-workers", type=int, default=0)
	parser.add_argument("--phase4-max-patients", type=int, default=0)
	parser.add_argument("--phase5-max-patients", type=int, default=0)
	args = parser.parse_args()

	REPORT_DIR.mkdir(parents=True, exist_ok=True)
	status = {
		"state": "started",
		"started_at": utc_now(),
		"clean_requested": bool(args.clean),
		"phase3_command": build_phase3_command(args),
		"phase4_command": build_phase4_command(args),
		"phase5_command": build_phase5_command(args),
	}
	write_status(status)

	log("Rerun pipeline starting")
	if args.clean:
		clean_outputs()

	phase3_log = REPORT_DIR / "rerun_phase3.log"
	phase4_log = REPORT_DIR / "rerun_phase4.log"
	phase5_log = REPORT_DIR / "rerun_phase5.log"

	phase3_command = build_phase3_command(args)
	phase4_command = build_phase4_command(args)
	phase5_command = build_phase5_command(args)

	stream_command(phase3_command, phase3_log)
	status["phase3"] = {"state": "completed", "completed_at": utc_now(), "log": str(phase3_log)}
	write_status(status)
	log(f"Phase 3 complete, checkpoint should be at {WORKSPACE_DIR / 'checkpoints' / 'phase3' / 'best.pth'}")

	if args.checkpoint == str(WORKSPACE_DIR / "checkpoints" / "phase3" / "best.pth"):
		args.checkpoint = str(WORKSPACE_DIR / "checkpoints" / "phase3" / "best.pth")
	stream_command(phase4_command, phase4_log)
	status["phase4"] = {"state": "completed", "completed_at": utc_now(), "log": str(phase4_log)}
	write_status(status)
	stream_command(phase5_command, phase5_log)
	status["phase5"] = {"state": "completed", "completed_at": utc_now(), "log": str(phase5_log)}
	status["state"] = "completed"
	status["completed_at"] = utc_now()
	write_status(status)

	log("Rerun pipeline complete")
	log(f"Phase 3 log: {phase3_log}")
	log(f"Phase 4 log: {phase4_log}")
	log(f"Phase 5 log: {phase5_log}")


if __name__ == "__main__":
	main()