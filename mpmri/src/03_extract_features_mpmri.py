import argparse
import json
import torch
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# --- PYTORCH 2.6 SECURITY PATCH ---
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# --- 1. CONFIGURATION & PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent  # mpmri/src
MPMRI_DIR = SCRIPT_DIR.parent                 # mpmri
ROOT_DIR = MPMRI_DIR.parent                   # prostate_mri_ensemble

TRAIN_IMG_DIR = MPMRI_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset501_ProstateMPMRI" / "imagesTr"
TEST_IMG_DIR  = MPMRI_DIR / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset501_ProstateMPMRI" / "imagesTs"
OUT_TRAIN_DIR = MPMRI_DIR / "data" / "03_frozen_features_mpmri" / "train"
OUT_TEST_DIR  = MPMRI_DIR / "data" / "03_frozen_features_mpmri" / "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Hardware locked: Using device {device}")

# --- 2. LINK PROFOUND ---
print("[*] Linking ProFound Architecture...")
PROFOUND_DIR = ROOT_DIR / "archive" / "ProFound"
sys.path.insert(0, str(PROFOUND_DIR))
from models.convnextv2 import convnextv2_tiny
sys.path.pop(0)

# --- 3. INITIALIZE PROFOUND (DYNAMIC SEARCH) ---
print("[*] Locating ProFound Weights...")
profound_ckpt_path = None

# Intelligently search the entire root workspace for the profound weights
for p in ROOT_DIR.rglob("*.pth"):
    if p.name.lower() == "profound.pth":
        profound_ckpt_path = p
        break

if not profound_ckpt_path:
    raise FileNotFoundError(
        f"Could not find 'profound.pth' anywhere inside {ROOT_DIR}. "
        "Please check if the file was deleted or moved outside the project."
    )

print(f"[*] Found weights at: {profound_ckpt_path}")
profound_model = convnextv2_tiny(in_chans=3) 
profound_ckpt = torch.load(profound_ckpt_path, map_location=device)
profound_model.load_state_dict(profound_ckpt["model"], strict=False)
profound_model.to(device).eval()

def extract_profound_spatial_features(profound_input: torch.Tensor) -> torch.Tensor:
    _, hidden_states = profound_model(profound_input, ret_hids=True)
    return hidden_states[-1]

def append_manifest_row(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# --- 4. THE EXTRACTION ENGINE ---
def process_patient(
    t2_img_path: Path,
    output_dir: Path,
    save_outputs: bool = True,
    print_shapes: bool = False,
    split_name: str = "train",
):
    patient_id = t2_img_path.name.replace("_0000.nii.gz", "")
    
    # Path setup for all 3 modalities
    adc_img_path = t2_img_path.parent / f"{patient_id}_0001.nii.gz"
    dwi_img_path = t2_img_path.parent / f"{patient_id}_0002.nii.gz"

    # Load T2, ADC, and DWI
    sitk_img_t2 = sitk.ReadImage(str(t2_img_path))
    img_array_t2 = sitk.GetArrayFromImage(sitk_img_t2)
    img_array_adc = sitk.GetArrayFromImage(sitk.ReadImage(str(adc_img_path)))
    img_array_dwi = sitk.GetArrayFromImage(sitk.ReadImage(str(dwi_img_path)))
    
    # Extract metadata based on T2
    original_shape = [int(x) for x in img_array_t2.shape]
    original_spacing = [float(x) for x in sitk_img_t2.GetSpacing()]
    original_origin = [float(x) for x in sitk_img_t2.GetOrigin()]
    original_direction = [float(x) for x in sitk_img_t2.GetDirection()]

    # ProFound Setup: Stacks T2, ADC, and DWI perfectly into a 3-channel (3, D, H, W) array
    stacked_array = np.stack([img_array_t2, img_array_adc, img_array_dwi], axis=0)
    profound_input = torch.from_numpy(stacked_array).float().unsqueeze(0).to(device)

    with torch.no_grad():
        profound_features = extract_profound_spatial_features(profound_input)

    profound_np = profound_features.squeeze().cpu().numpy()

    if print_shapes:
        print(f"[{patient_id}] profound spatial shape: {profound_np.shape}")

    patient_meta = {
        "patient_id": patient_id,
        "split": split_name,
        "source_image": str(t2_img_path),
        "native_image": {
            "shape_dhw": original_shape,
            "spacing": original_spacing,
            "origin": original_origin,
            "direction": original_direction,
        },
        "tensors": {
            "profound_input_shape_bcdhw": [int(x) for x in profound_input.shape],
            "profound_output_shape_cdhw": [int(x) for x in profound_np.shape],
        },
        "files": {
            "profound": f"{patient_id}_profound.npy",
            "meta": f"{patient_id}_meta.json",
        },
    }

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{patient_id}_profound.npy", profound_np)
        with open(output_dir / f"{patient_id}_meta.json", "w", encoding="utf-8") as f:
            json.dump(patient_meta, f, indent=2)
        append_manifest_row(output_dir / "manifest.jsonl", patient_meta)


def process_folder(input_dir: Path, output_dir: Path, split_name: str):
    image_files = sorted(input_dir.glob("*_0000.nii.gz"))
    if not image_files:
        return

    for t2_img_path in tqdm(image_files, desc=f"Processing {input_dir.name}"):
        try:
            process_patient(t2_img_path, output_dir, save_outputs=True, split_name=split_name)
        except Exception as exc:
            patient_id = t2_img_path.name.replace("_0000.nii.gz", "")
            print(f"[!] Skipping {patient_id} due to error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Extract frozen ProFound mpMRI features.")
    parser.add_argument("--dry-run", action="store_true", help="Process exactly one patient and only print tensor shapes.")
    args = parser.parse_args()

    if args.dry_run:
        target_paths = sorted(TRAIN_IMG_DIR.glob("*_0000.nii.gz"))
        if not target_paths:
            raise FileNotFoundError(f"No _0000.nii.gz files found in {TRAIN_IMG_DIR}")
        
        print("\n--- Dry Run: one patient only ---")
        process_patient(target_paths[0], OUT_TRAIN_DIR, save_outputs=False, print_shapes=True, split_name="train")
        print("--- Dry Run Complete ---")
        return

    print("\n--- Initiating GPU Feature Extraction Pipeline ---")
    process_folder(TRAIN_IMG_DIR, OUT_TRAIN_DIR, split_name="train")
    process_folder(TEST_IMG_DIR, OUT_TEST_DIR, split_name="test")
    print("\n--- Extraction Complete! ---")


if __name__ == "__main__":
    main()