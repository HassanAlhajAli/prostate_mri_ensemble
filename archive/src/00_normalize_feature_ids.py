import json
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = WORKSPACE_DIR / "data" / "02_frozen_features"


def canonical_id(pid: str) -> str:
    return pid[:-5] if pid.endswith("_0000") else pid


def update_meta_content(meta: dict, old_id: str, new_id: str) -> dict:
    meta["patient_id"] = new_id
    files = meta.get("files", {})
    files["profound"] = f"{new_id}_profound.npy"
    files["sam_embedding"] = f"{new_id}_sam_embedding.npy"
    files["sam_logits"] = f"{new_id}_sam_logits.npy"
    files["meta"] = f"{new_id}_meta.json"
    meta["files"] = files
    source_image = meta.get("source_image")
    if isinstance(source_image, str):
        meta["source_image"] = source_image.replace(old_id, new_id)
    return meta


def rename_if_exists(old_path: Path, new_path: Path) -> bool:
    if not old_path.exists():
        return False
    if old_path == new_path:
        return False
    if new_path.exists():
        return False
    old_path.rename(new_path)
    return True


def normalize_split(split_dir: Path):
    migrated = 0
    skipped_existing = 0

    meta_paths = sorted(split_dir.glob("*_meta.json"))
    rebuilt_records = []

    for meta_path in meta_paths:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        old_id = str(meta.get("patient_id", meta_path.name[: -len("_meta.json")]))
        new_id = canonical_id(old_id)

        old_files = {
            "profound": split_dir / f"{old_id}_profound.npy",
            "sam_embedding": split_dir / f"{old_id}_sam_embedding.npy",
            "sam_logits": split_dir / f"{old_id}_sam_logits.npy",
            "meta": split_dir / f"{old_id}_meta.json",
        }
        new_files = {
            "profound": split_dir / f"{new_id}_profound.npy",
            "sam_embedding": split_dir / f"{new_id}_sam_embedding.npy",
            "sam_logits": split_dir / f"{new_id}_sam_logits.npy",
            "meta": split_dir / f"{new_id}_meta.json",
        }

        if old_id != new_id:
            for key in ["profound", "sam_embedding", "sam_logits", "meta"]:
                changed = rename_if_exists(old_files[key], new_files[key])
                if changed:
                    migrated += 1
                elif old_files[key].exists() and new_files[key].exists():
                    skipped_existing += 1

        updated_meta = update_meta_content(meta, old_id, new_id)
        target_meta_path = new_files["meta"]
        with open(target_meta_path, "w", encoding="utf-8") as f:
            json.dump(updated_meta, f, indent=2)

        rebuilt_records.append(updated_meta)

    rebuilt_records = sorted(rebuilt_records, key=lambda r: str(r.get("patient_id", "")))
    manifest_path = split_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in rebuilt_records:
            f.write(json.dumps(rec) + "\n")

    return {
        "split": split_dir.name,
        "meta_records": len(rebuilt_records),
        "renamed_files": migrated,
        "skipped_existing": skipped_existing,
    }


def main():
    summaries = []
    for split in ["train", "test"]:
        split_dir = FEATURES_DIR / split
        if split_dir.exists():
            summaries.append(normalize_split(split_dir))

    print("Normalization summary:")
    for s in summaries:
        print(
            f"- {s['split']}: meta_records={s['meta_records']}, "
            f"renamed_files={s['renamed_files']}, skipped_existing={s['skipped_existing']}"
        )


if __name__ == "__main__":
    main()
