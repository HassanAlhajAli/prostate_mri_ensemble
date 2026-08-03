import os
import shutil
import random
import json
import logging
from pathlib import Path
from typing import List, Tuple
import SimpleITK as sitk
from tqdm import tqdm

# --- 1. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- 2. CONFIGURATION CONSTANTS ---
WORKSPACE_DIR = Path("/mnt/c/Disk D/UCL MSc/Thesis/prostate_mri_ensemble")
RAW_DATA_DIR = WORKSPACE_DIR / "data" / "01_promis_raw" / "promis_mapped"

# Point to the NEW isolated data folder
MPMRI_WORKSPACE = WORKSPACE_DIR / "mpmri"
NNUNET_DATASET_DIR = MPMRI_WORKSPACE / "data" / "nnUNet_data" / "nnUNet_raw" / "Dataset501_ProstateMPMRI"

TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
PATIENT_PREFIX = "P-"

# --- 3. HELPER FUNCTIONS ---
def create_nnunet_dirs(base_dir: Path) -> None:
    dirs_to_make = ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]
    for d in dirs_to_make:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    logging.info("nnU-Net directories created successfully.")

def get_patient_folders(data_dir: Path, prefix: str) -> List[str]:
    patients = [
        f for f in os.listdir(data_dir) 
        if os.path.isdir(data_dir / f) and f.startswith(prefix)
    ]
    logging.info(f"Found {len(patients)} total patients matching prefix '{prefix}'.")
    return patients

def split_dataset(patients: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    random.shuffle(patients)
    
    split_idx = int(len(patients) * train_ratio)
    train_patients = patients[:split_idx]
    test_patients = patients[split_idx:]
    
    logging.info(f"Allocated {len(train_patients)} to Training ({train_ratio*100}%) and {len(test_patients)} to Test Lockbox.")
    return train_patients, test_patients

def copy_patient_files(patient_list: List[str], is_train: bool, raw_dir: Path, target_dir: Path) -> None:
    img_folder = "imagesTr" if is_train else "imagesTs"
    lbl_folder = "labelsTr" if is_train else "labelsTs"
    
    missing_data_count = 0
    phase_name = "Training Data" if is_train else "Test Data"

    for pid in tqdm(patient_list, desc=f"Processing {phase_name}", unit="scan"):
        src_t2 = raw_dir / pid / "t2.nii.gz"
        src_adc = raw_dir / pid / "adc.nii.gz"
        src_dwi = raw_dir / pid / "dwi.nii.gz"
        src_lbl = raw_dir / pid / "lesion_ordered.nii.gz"
        
        dst_t2 = target_dir / img_folder / f"{pid}_0000.nii.gz"
        dst_adc = target_dir / img_folder / f"{pid}_0001.nii.gz"
        dst_dwi = target_dir / img_folder / f"{pid}_0002.nii.gz"
        dst_lbl = target_dir / lbl_folder / f"{pid}.nii.gz"
        
        # Ensure all modalities exist before copying
        if src_t2.exists() and src_adc.exists() and src_dwi.exists() and src_lbl.exists():
            shutil.copy(src_t2, dst_t2)
            shutil.copy(src_adc, dst_adc)
            shutil.copy(src_dwi, dst_dwi)
            
            lbl_img = sitk.ReadImage(str(src_lbl))
            lbl_arr = sitk.GetArrayFromImage(lbl_img)
            lbl_arr[lbl_arr > 0] = 1  
            
            binarized_img = sitk.GetImageFromArray(lbl_arr)
            binarized_img.CopyInformation(lbl_img) 
            sitk.WriteImage(binarized_img, str(dst_lbl))
        else:
            missing_data_count += 1
            
    if missing_data_count > 0:
        logging.warning(f"Skipped {missing_data_count} patients due to missing MRI sequences or lesion files.")

def generate_dataset_json(target_dir: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {
            "0": "T2",
            "1": "ADC",
            "2": "DWI"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    json_path = target_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    logging.info(f"Generated dataset.json at {json_path}")

# --- 4. MAIN EXECUTION PIPELINE ---
def main() -> None:
    logging.info("Starting isolated multi-parametric data split pipeline...")
    create_nnunet_dirs(NNUNET_DATASET_DIR)
    
    patient_folders = get_patient_folders(RAW_DATA_DIR, PATIENT_PREFIX)
    train_patients, test_patients = split_dataset(patient_folders, TRAIN_SPLIT_RATIO, RANDOM_SEED)
    
    copy_patient_files(train_patients, is_train=True, raw_dir=RAW_DATA_DIR, target_dir=NNUNET_DATASET_DIR)
    copy_patient_files(test_patients, is_train=False, raw_dir=RAW_DATA_DIR, target_dir=NNUNET_DATASET_DIR)
    
    generate_dataset_json(NNUNET_DATASET_DIR, num_training=len(train_patients))
    logging.info("Split, binarization, and formatting complete!")

if __name__ == "__main__":
    main()
