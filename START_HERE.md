# Research handoff: start here

This repository contains the completed CLIP attribution + SAM research project. Continue from the saved experiments; do not restart or silently replace them.

## Read in order

1. `SMART_SAM_RESULTS.md`: current result and its limitations.
2. `SMART_SAM_REPRODUCTION.md`: implementation changes, commands and environment details.
3. `outputs/smart_sam/frozen/FROZEN_METHOD.md` and `model_files/models.json`: frozen method and fitted parameters.
4. `outputs/smart_sam/heldout/HELDOUT_REPORT.md`: complete prospective evaluation.
5. `outputs/smart_sam/development/DEVELOPMENT_REPORT.md`: grouped validation and rejected alternatives.
6. `SAM_DIAGNOSTIC_RESULTS.md`, `COCO_SAM_RESULTS.md`, `REFINEMENT_RESULTS.md`, `REPRODUCTION_NOTES.md`: earlier findings and mathematical ambiguities.

## Current scientific state

The original 500 COCO images / 1,000 target pairs are DEVELOPMENT ONLY. They informed diagnostics, feature design, fitting and calibration. Five outer image folds and four inner folds cross-fit the selector for gate training.

A nine-feature standardized linear pairwise logistic ranker selects one of three masks from unchanged SAM ViT-B, prompted by three positive CLIP-attribution points. A log attribution-density threshold of 0.72536122868084263 decides whether to use that mask or retain the original CLIP binary mask. CLIP and SAM weights were not trained. Features, fitted parameters and source hashes were frozen before the second cohort was selected.

The second cohort contains 500 image-disjoint COCO val2017 images / 1,000 pairs, seed 2026. Heldout mIoU: CLIP 0.2699; naive SAM-score selection 0.3949; smart selection 0.4291; frozen smart+density fallback 0.4335. Final versus naive gain +0.0386, paired 95% CI [0.0274, 0.0497]. Harm falls 41.1% -> 26.2%. Semantic success 73.2% -> 73.7% and precise-wrong-object rate 7.8% -> 7.4% do NOT establish significant improvements over naive SAM. Wrong-object rate remains above CLIP's 1.5%. In 49.2% of pairs no candidate reaches IoU 0.5. Combined candidate+fallback oracle: 0.5181.

The heldout cohort has now been evaluated and inspected. Never tune on it and still call that tuning prospective heldout evidence. A future modified method needs a separately defined evaluation plan and untouched test data. No next experiment is authorized by this handoff alone; follow the next user's request.

## What this Git snapshot contains

Code, tests, configurations, all root research notes, structured experiment reports/metrics/manifests, frozen fitted models, hashes and 24 current heldout example panels. Most older galleries retain their index/manifest but their large image collections are not included. Full ImageNet per-image checkpoint metadata is omitted; aggregate and per-image metric tables are retained.

Datasets, downloaded archives, CLIP/SAM weight files, saved attribution tensors, SAM candidate arrays and most generated images are excluded from Git. Therefore a clone is sufficient to understand the study, inspect fitted models and analyze saved CSVs, but is NOT a complete raw-artifact backup. See `ARTIFACTS.md`. Do not claim all original integrity checks can run from a clone without restoring excluded artifacts.

Original workspace: `C:/CREMI/trdp/code/cda_clip_reproduction`; environment: `C:/CREMI/trdp/.venv`. These paths are historical, not portable defaults. Python 3.11.16, torch 2.11.0+cu128 and torchvision 0.26.0+cu128 were used on Windows with an RTX 4000 Ada. Version files: `requirements*.txt`, `environment.txt`, heldout `run.json`. Do not commit a virtual environment. Review platform-specific dependencies before installing on another OS.

## Where to implement

- `scripts/smart_sam_core.py`: inference-only features, linear ranker, comparison gates.
- `scripts/develop_smart_sam.py`: old-candidate extraction and nested image-grouped CV.
- `scripts/freeze_smart_sam.py`: fitted method and source/checkpoint hash seal.
- `scripts/select_smart_sam_heldout.py`: post-freeze eligibility and disjoint cohort.
- `scripts/run_smart_sam_heldout.py`: RGB/text-only inference, no GT evaluation.
- `scripts/evaluate_smart_sam.py`, `report_smart_sam.py`, `visualize_smart_sam.py`, `audit_smart_sam.py`: separate evaluation and reporting.
- `src/cda_clip.py`, `src/refinement.py`, `scripts/sam_inference_core.py`: previously frozen attribution and SAM geometry; preserve them.

At handoff, 41 tests passed. Previous output files and all frozen source hashes were verified unchanged. A fresh clone has fewer raw artifacts by design; missing files are not evidence that the original run failed.

## Prompt for a new Codex session

> Read START_HERE.md, SMART_SAM_RESULTS.md, SMART_SAM_REPRODUCTION.md and the frozen method before doing anything. This is an existing completed research project. Preserve the frozen experiments and negative findings. Explain the current state and identify which raw artifacts are available locally. Wait for my next research instruction; do not retrain, retune on the evaluated heldout cohort, or regenerate existing results automatically.
