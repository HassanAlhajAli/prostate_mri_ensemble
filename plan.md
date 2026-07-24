# Clean Workspace Plan: nnU-Net + ProFound Only

# Minimal Active Workspace Plan

## Active Workspace Rules
Keep only the files and folders needed for the current nnU-Net baseline and the fixed data split. Everything else should be archived.

## Keep Active
* `plan.md`
* `src/01_data_split.py`
* `data/01_promis_raw/`
* `data/02_frozen_features/`
* `data/nnUNet_data/`
* `checkpoint_best.pth`
* `checkpoint_final.pth`
* `training_log_2026_7_23_11_37_50.txt`

## Archive Everything Else
* All remaining scripts in `src/`
* ProFound source and model files
* SAM-Med3D source, scripts, reports, and visualizations
* `reports/` contents other than historical archive material
* `checkpoints/` contents other than the two kept nnU-Net checkpoints
* exploratory notebooks, temporary logs, and installer files

## Current Workflow
* The split is already complete and remains the fixed source of truth.
* nnU-Net training has already been run on the remote GPU.
* The next work should use the kept split and the kept nnU-Net artifacts only.
* Any future ProFound retraining should start from scratch in a clean, separate path if needed.

## Phase 0: Archive Deprecated / Exploratory Material [In Progress]
* Move SAM-Med3D scripts, SAM-Med3D reports, SAM-Med3D visualizations, and the SAM-Med3D vendor tree into `archive/`.
* Move one-off notebook or scratch exploration files into `archive/`.
* Move temporary rerun logs or stale experiment outputs into `archive/` if they are no longer part of the live workflow.
* Keep the archived material readable and complete so old experiments remain reproducible.

## Phase 1: Data Preparation [Complete]
* The dataset split is already complete.
* Keep the split as the fixed source of truth for all future retraining and evaluation.

## Phase 2: nnU-Net Baseline Setup & Evaluation
* Keep nnU-Net isolated in its own folder structure.
* Run inference on the test set using the trained nnU-Net weights.
* Record the standalone nnU-Net results in its own report.

## Phase 3: ProFound Retraining From Scratch
* Rebuild the ProFound path from scratch rather than relying on old Phase 3 outputs.
* Keep only the ProFound retraining code that is still needed.
* Regenerate any required ProFound features or support artifacts using the fixed split.
* Evaluate ProFound standalone performance on the test set.

## Phase 4: Ensemble Fusion & Comparison
* Use the standalone nnU-Net and ProFound outputs as the inputs to any future fusion.
* Compare Simple Averaging, DST, and LoMix only after the standalone baselines are finalized.
* Map predictions back to native geometry and evaluate final ensemble performance against both baselines.