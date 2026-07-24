import os
import sys
import argparse
import json
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# --- PYTORCH 2.6 SECURITY PATCH ---
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# --- 1. CONFIGURATION & PATH SETUP ---
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROFOUND_DIR = WORKSPACE_DIR / "ProFound"
SAM_MED3D_DIR = WORKSPACE_DIR / "SAM-Med3D"

TRAIN_IMG_DIR = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "imagesTr"
TEST_IMG_DIR  = WORKSPACE_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset500_PROMIS" / "imagesTs"
OUT_TRAIN_DIR = WORKSPACE_DIR / "data" / "02_frozen_features" / "train"
OUT_TEST_DIR  = WORKSPACE_DIR / "data" / "02_frozen_features" / "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Hardware locked: Using device {device}")

# --- 2. DODGING THE IMPORT TRAP ---
print("[*] Linking Foundation Model Architectures...")

sys.path.insert(0, str(PROFOUND_DIR))
from models.convnextv2 import convnextv2_tiny
sys.path.pop(0)

sys.path.insert(0, str(SAM_MED3D_DIR))
from segment_anything.build_sam3D import build_sam3D_vit_b_ori
sys.path.pop(0)

# --- 3. INITIALIZE THE BRAINS ---
print("[*] Loading Neural Network Weights into VRAM...")


def load_sam_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned_state_dict[key[len("module."):]] = value
            else:
                cleaned_state_dict[key] = value
        state_dict = cleaned_state_dict

    return state_dict


def center_crop_or_pad_3d(volume: torch.Tensor, target_size: int):
    depth, height, width = volume.shape[-3:]
    original_shape = [int(depth), int(height), int(width)]

    pad_depth = max(target_size - depth, 0)
    pad_height = max(target_size - height, 0)
    pad_width = max(target_size - width, 0)

    pad_depth_before = pad_depth // 2
    pad_depth_after = pad_depth - pad_depth_before
    pad_height_before = pad_height // 2
    pad_height_after = pad_height - pad_height_before
    pad_width_before = pad_width // 2
    pad_width_after = pad_width - pad_width_before

    if pad_depth or pad_height or pad_width:
        volume = F.pad(
            volume,
            (
                pad_width_before,
                pad_width_after,
                pad_height_before,
                pad_height_after,
                pad_depth_before,
                pad_depth_after,
            ),
        )

    depth, height, width = volume.shape[-3:]
    padded_shape = [int(depth), int(height), int(width)]
    start_depth = max((depth - target_size) // 2, 0)
    start_height = max((height - target_size) // 2, 0)
    start_width = max((width - target_size) // 2, 0)

    volume = volume[
        ...,
        start_depth:start_depth + target_size,
        start_height:start_height + target_size,
        start_width:start_width + target_size,
    ]

    transform_meta = {
        "original_shape": original_shape,
        "target_shape": [int(target_size), int(target_size), int(target_size)],
        "pad_before": [int(pad_depth_before), int(pad_height_before), int(pad_width_before)],
        "pad_after": [int(pad_depth_after), int(pad_height_after), int(pad_width_after)],
        "padded_shape": padded_shape,
        "crop_start": [int(start_depth), int(start_height), int(start_width)],
        "crop_end": [int(start_depth + target_size), int(start_height + target_size), int(start_width + target_size)],
        "cropped_shape": [int(target_size), int(target_size), int(target_size)],
    }
    return volume, transform_meta


def normalize_volume_for_sam(volume: torch.Tensor) -> torch.Tensor:
    foreground = volume != 0
    values = volume[foreground] if foreground.any() else volume
    mean = values.mean()
    std = values.std().clamp_min(1e-6)
    return (volume - mean) / std


def prepare_sam_input(img_array: np.ndarray, target_size: int):
    volume = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
    volume, transform_meta = center_crop_or_pad_3d(volume, target_size)
    volume = normalize_volume_for_sam(volume)
    return volume.to(device), transform_meta


def append_manifest_row(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def extract_sam_outputs(sam_input_tensor: torch.Tensor):
    image_embeddings = sam_model.image_encoder(sam_input_tensor)
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
    low_res_logits, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return image_embeddings, low_res_logits, iou_predictions


def extract_profound_spatial_features(profound_input: torch.Tensor) -> torch.Tensor:
    _, hidden_states = profound_model(profound_input, ret_hids=True)
    return hidden_states[-1]

# 1. Boot up ProFound
profound_model = convnextv2_tiny(in_chans=3) 
profound_ckpt = torch.load(WORKSPACE_DIR / "checkpoints" / "profound.pth", map_location=device)
profound_model.load_state_dict(profound_ckpt["model"], strict=False)
profound_model.to(device).eval()

# 2. Boot up SAM-Med3D
# Build the 3D ViT-B original body that matches the turbo checkpoint.
sam_model = build_sam3D_vit_b_ori(checkpoint=None)

# Unpack the checkpoint wrapper, then load the actual state dict into the model.
sam_weights = load_sam_checkpoint(WORKSPACE_DIR / "checkpoints" / "sam_med3d_turbo.pth")
missing_keys, unexpected_keys = sam_model.load_state_dict(sam_weights, strict=False)
print(f"[*] SAM-Med3D load summary: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
if missing_keys:
    print(f"[*] First missing keys: {missing_keys[:10]}")
if unexpected_keys:
    print(f"[*] First unexpected keys: {unexpected_keys[:10]}")
sam_model.to(device).eval()

# --- 4. THE EXTRACTION ENGINE ---
def process_patient(
    img_path: Path,
    output_dir: Path,
    save_outputs: bool = True,
    print_shapes: bool = False,
    split_name: str = "train",
):
    if img_path.name.endswith("_0000.nii.gz"):
        patient_id = img_path.name[: -len("_0000.nii.gz")]
    else:
        patient_id = img_path.name.replace(".nii.gz", "")

    sitk_img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(sitk_img)
    original_shape = [int(x) for x in img_array.shape]
    original_spacing = [float(x) for x in sitk_img.GetSpacing()]
    original_origin = [float(x) for x in sitk_img.GetOrigin()]
    original_direction = [float(x) for x in sitk_img.GetDirection()]

    raw_input_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0).to(device)
    sam_input_tensor, sam_transform_meta = prepare_sam_input(img_array, sam_model.image_encoder.img_size)

    profound_input = raw_input_tensor.repeat(1, 3, 1, 1, 1)

    with torch.no_grad():
        profound_features = extract_profound_spatial_features(profound_input)
        sam_embeddings, sam_logits, _ = extract_sam_outputs(sam_input_tensor)

    profound_np = profound_features.squeeze().cpu().numpy()
    sam_embeddings_np = sam_embeddings.squeeze().cpu().numpy()
    sam_logits_np = sam_logits.squeeze().cpu().numpy()

    if print_shapes:
        print(f"[{patient_id}] profound spatial shape: {profound_np.shape}")
        print(f"[{patient_id}] sam embedding shape: {sam_embeddings_np.shape}")
        print(f"[{patient_id}] sam logits shape: {sam_logits_np.shape}")

    patient_meta = {
        "patient_id": patient_id,
        "split": split_name,
        "source_image": str(img_path),
        "native_image": {
            "shape_dhw": original_shape,
            "spacing": original_spacing,
            "origin": original_origin,
            "direction": original_direction,
        },
        "sam_preprocess": sam_transform_meta,
        "tensors": {
            "profound_input_shape_bcdhw": [int(x) for x in profound_input.shape],
            "sam_input_shape_bcdhw": [int(x) for x in sam_input_tensor.shape],
            "profound_output_shape_cdhw": [int(x) for x in profound_np.shape],
            "sam_embedding_shape_cdhw": [int(x) for x in sam_embeddings_np.shape],
            "sam_logits_shape_dhw": [int(x) for x in sam_logits_np.shape],
        },
        "files": {
            "profound": f"{patient_id}_profound.npy",
            "sam_embedding": f"{patient_id}_sam_embedding.npy",
            "sam_logits": f"{patient_id}_sam_logits.npy",
            "meta": f"{patient_id}_meta.json",
        },
    }

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{patient_id}_profound.npy", profound_np)
        np.save(output_dir / f"{patient_id}_sam_embedding.npy", sam_embeddings_np)
        np.save(output_dir / f"{patient_id}_sam_logits.npy", sam_logits_np)
        with open(output_dir / f"{patient_id}_meta.json", "w", encoding="utf-8") as f:
            json.dump(patient_meta, f, indent=2)
        append_manifest_row(output_dir / "manifest.jsonl", patient_meta)


def process_folder(input_dir: Path, output_dir: Path, split_name: str):
    image_files = sorted(input_dir.glob("*.nii.gz"))
    if not image_files:
        return

    for img_path in tqdm(image_files, desc=f"Processing {input_dir.name}"):
        try:
            process_patient(img_path, output_dir, save_outputs=True, split_name=split_name)
        except Exception as exc:
            patient_id = img_path.name.replace(".nii.gz", "")
            print(f"[!] Skipping {patient_id} due to error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Extract frozen ProFound and SAM-Med3D features.")
    parser.add_argument("--dry-run", action="store_true", help="Process exactly one patient and only print tensor shapes.")
    parser.add_argument(
        "--dry-run-patient",
        type=str,
        default=None,
        help="Patient ID to use for dry-run mode. Defaults to the first patient in the train split.",
    )
    args = parser.parse_args()

    if args.dry_run:
        if args.dry_run_patient is not None:
            candidate_paths = list(TRAIN_IMG_DIR.glob(f"{args.dry_run_patient}*.nii.gz"))
            if not candidate_paths:
                raise FileNotFoundError(f"Could not find a train image for patient '{args.dry_run_patient}'.")
            target_path = sorted(candidate_paths)[0]
        else:
            target_paths = sorted(TRAIN_IMG_DIR.glob("*.nii.gz"))
            if not target_paths:
                raise FileNotFoundError(f"No .nii.gz files found in {TRAIN_IMG_DIR}")
            target_path = target_paths[0]

        print("\n--- Dry Run: one patient only ---")
        process_patient(target_path, OUT_TRAIN_DIR, save_outputs=False, print_shapes=True, split_name="train")
        print("--- Dry Run Complete ---")
        return

    print("\n--- Initiating GPU Feature Extraction Pipeline ---")
    process_folder(TRAIN_IMG_DIR, OUT_TRAIN_DIR, split_name="train")
    process_folder(TEST_IMG_DIR, OUT_TEST_DIR, split_name="test")
    print("\n--- Extraction Complete! ---")


if __name__ == "__main__":
    main()