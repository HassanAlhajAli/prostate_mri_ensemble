# Clean Workspace Plan: Multi-Modal Ensemble & Hypernetwork Fusion

## Active Workspace Rules
Keep only the files and folders needed for the active baseline pipelines, cached predictions, and the fixed data split. Everything else should be archived.

## Keep Active
* `plan.md`
* `src/` (Active fusion & hypernetwork modules)
* `data/01_promis_raw/`
* `data/02_frozen_features/`
* `data/nnUNet_data/`
* `checkpoints/nnunet_mpmri/`
* `checkpoints/profound_mpmri/`
* `checkpoints/hyper_lomix_mpmri/`
* `reports/ensemble_cache_mpmri/`
* `training_log_*.txt`

## Archive Everything Else
* Stale scripts and deprecated experiment variations
* SAM-Med3D source, scripts, reports, and visualizations
* Scratch exploratory notebooks and temporary test output logs
* Unused intermediate checkpoints outside active best/final weights

## Current Workflow
* The dataset split remains fixed as the single source of truth across all modalities and models.
* Standalone base model probability maps (nnU-Net and ProFound) are cached as `.npy` arrays to prevent redundant compute.
* Hyper-LoMix uses dynamic weight generation based on tabular clinical context to fuse base model predictions.

---

## Phase 0: Archive Deprecated / Exploratory Material [Complete]
* Moved legacy exploration notebooks, SAM-Med3D scripts, and temporary rerun logs into `archive/`.
* Preserved read-only access to historical logs and raw features for reproducibility.

## Phase 1: Data Preparation & Fixed Split [Complete]
* Fixed the train/val/test splits across PROMIS cases.
* Standardized ground truth label access across all downstream training scripts.

## Phase 2: nnU-Net Baseline Setup & Evaluation [Complete]
* Trained/evaluated standalone nnU-Net on the fixed split.
* Preserved model checkpoints (`checkpoint_final.pth`) and extracted standalone evaluation metrics.

## Phase 3: ProFound Standalone Retraining & Feature Extraction [Complete]
* Retrained the 3D ProFound decoder path on frozen feature embeddings.
* Preserved champion weights (`final_mpmri_champion.pt`) and computed standalone performance metrics.

## Phase 4: Standard Ensemble Fusion & Baseline Comparison [Complete]
* Implemented rule-based fusion baselines: Boolean AND, Boolean OR, Simple Averaging, and Dempster-Shafer Theory (DST).
* Implemented standard LoMix static spatial fusion model.
* Generated baseline leaderboard comparing all standard fusion strategies against standalone base models.

---

## Phase 5: mpMRI Extension & Prediction Caching [Complete]
* Extended the single-modality baseline framework to multi-parametric MRI inputs (T2W, ADC, DWI).
* Cached 3D soft probability volumes (`.npy`) for both nnU-Net and ProFound across train and test splits in `reports/ensemble_cache_mpmri/`.
* Verified zero data duplication by reading directly from cached probability arrays during downstream ensemble passes.

## Phase 6: Hypernetwork-Guided Dynamic Fusion (Hyper-LoMix) [In Progress]
* **Clinical Meta-Feature Extraction:**
  * Parse tabular clinical metadata (`lesion_ordered.csv`).
  * Extract patient-level severity features (e.g., 2D mode: max ISUP score, max PI-RADS score per patient) and scale to [0, 1].
* **Hypernetwork Architecture Implementation:**
  * Construct dual-pathway architecture (`HyperLoMixFusionNet`).
  * **Spatial Body:** Processes 3D probability stacks from nnU-Net and ProFound.
  * **Clinical Brain (Hypernetwork):** MLP that ingests patient clinical vectors to dynamically generate weights and biases for the spatial fusion convolutions.
* **Training & Optimization:**
  * Train the dynamic kernel parameters using mixed precision (`autocast`) with combined BCE, Soft Dice, and Focal loss functions.
  * Optimize threshold selection on the validation set during training iterations.
* **Test Set Evaluation & Leaderboard Generation:**
  * Run inference on unseen test cases using the champion dynamic model (`hyper_lomix_best.pt`).
  * Map predicted masks back to native NIfTI geometry.
  * Compare Hyper-LoMix against standalone models (nnU-Net, ProFound), classical fusion (AND, OR, Average, DST), and static LoMix inside the final `hyper_lomix_test_metrics.json` leaderboard.