# Current state: all six external-source inference and evaluation cells complete

RefCOCO, RefCOCO+ and RefCOCOg CS/GE predictions have all been evaluated from saved arrays. No inference rerun or tuning was performed for evaluation. RefCOCO/RefCOCO+ results are in REFCOCO_AND_REFCOCOPLUS_SHARE_SUMMARY.txt; RefCOCOg full and predefined strict-subset statistics are in outputs/cross_attribution_generalization/refcocog_saved_source_statistics.json and companion CSV files. The frozen smart selector has positive paired IoU intervals for both sources on all three datasets. Preserve source-specific fallback tradeoffs. The broader matrix report/final replay audit has not been run; do not imply it has.

The user authorized GitHub backup on October 5, 2026. Publication requires restored GitHub authentication; consult BACKUP_STATUS.md. New raw cross-attribution outputs are archived separately at C:/CREMI/trdp/research-backup-2026-10-05. Do not restart inference.

# Earlier authorized work: frozen cross-product evaluation

The user authorized CS and GE on the exact existing RefCOCO, RefCOCO+ and RefCOCOg manifests. See CROSS_ATTRIBUTION_PLAN.md, CROSS_ATTRIBUTION_REPRODUCTION.md and outputs/cross_attribution_generalization/. Inference is in progress via scripts/run_cross_attribution_matrix.py. Never restart or overwrite completed cells; inspect each run.json and resume only with identical signatures. No GT scoring until all six inference cells terminate. Existing CDA/dense results are reused. The earlier STOP statements below predate this explicit authorization. Stop after the final matrix and audit.

The earlier Git backup commit is b418a6a4; remote push/release upload has not yet been confirmed. Local verified output archives are in C:/CREMI/trdp/research-backup-2026-10-01; see BACKUP_STATUS.md.

# Research handoff: start here

This repository contains the completed CLIP attribution + SAM research project. Continue from the saved experiments; do not restart or silently replace them.

## Latest external baseline: official Grad-ECLIP and frozen CASR transfer

Read `GRAD_ECLIP_BASELINE_REPORT.md`, `GRAD_ECLIP_REPRODUCTION.md` and `outputs/external_baselines/grad_eclip/`. Official Cyang-Zhao/Grad-Eclip commit `e370e6cb194faf2020f5d1ed268f9d57e91a38e6`; original image-demo method cells executed verbatim, OpenAI ViT-B/16, n=1, native full-image/patch-compatible preprocessing and half precision. Official demo and prompt-sensitivity checks passed. No official GE SAM integration is supplied; all GE SAM rows are common POS3/ViT-B experiments. No method fitting or modification.

Same 500 COCO images / 1,000 targets: GE map mIoU **0.2683**, naive SAM **0.3972**, frozen smart **0.4365**, final CASR **0.4411**. Smart-minus-naive **+0.0393 [0.0283, 0.0502]**; Dice **+0.0428 [0.0322, 0.0537]**. Fallback-minus-smart IoU **+0.0046 [0.0011, 0.0077]**. Harm versus GE_MAP **41.2% -> 32.2% -> 26.5%**; fallback harm difference -5.7 percentage points [-7.1, -4.3]. Candidate accuracy **44.0% -> 57.3%**, difference +13.3 points [9.6, 17.0]; regret **0.0969 -> 0.0576**; oracle **0.4941**, no candidate at IoU>=0.5 in **48.0%** of pairs.

Semantic-success proxy **74.2% -> 74.7% -> 75.4%**; neither smart-minus-naive nor fallback-minus-smart interval establishes a change. Precise-wrong-object rates **8.4% -> 8.0% -> 7.4%**. GE map PG0.6930, EPG0.3806, AP0.5445, PAcc0.7371. Retain raw native attribution scale; no CDA/CS map normalization was added.

Cross-source map/naive/smart/final mIoU: CDA **0.2699/0.3949/0.4291/0.4335**; CS common **0.2997/0.4295/0.4648/0.4684**; GE **0.2683/0.3972/0.4365/0.4411**. Official CS ViT-H 0.5000 remains separate context. This supports frozen selector transfer to two external sources, with source-specific fallback/semantic tradeoffs, not universal superiority. Different upstream views/resolutions and one already-inspected selected cohort limit claims.

69 tests passed. All 1,000 GE pairs retained, zero/constant maps and empty candidates all zero. 24 panels across eight requested groups, full CIs/CSV/JSON, raw maps/candidates/features and source/checkpoint hashes retained; final audit in `outputs/external_baselines/grad_eclip/audit.json`. Inference including smoke/setup/hash I/O 332.9s; evaluation12.3s. Main environment, frozen CASR and all earlier experiments preserved. New studies remain local, not pushed.

STOP external-attribution experiments here as requested. Results support drafting a bounded paper narrative, but do not authorize new paper experiments, datasets, methods, tuning or GitHub push. Do not rerun previous inference to reconstruct missing Git-excluded raw artifacts; consult ARTIFACTS.md.

## Earlier external baseline: official CLIP Surgery on fixed dense COCO

Read `CLIP_SURGERY_BASELINE_REPORT.md`, `CLIP_SURGERY_REPRODUCTION.md` and the full artifacts under `outputs/external_baselines/clip_surgery/`. Official repository commit `d4696d47f49cfe70f49140afe5eb94f94c5f59bc`; complete unmodified demo passed. Exact previous heldout 500 images / 1,000 targets, no tuning or refitting. This cohort was already inspected before this comparison.

Official Text2Points/SAM ViT-H mIoU **0.5000**, higher than existing CASR **0.4335**, paired difference **+0.0665 [0.0473, 0.0865]**. Different SAM backbone, prompts, full-image view and template ensemble prevent attributing this difference to the map formula alone. Under shared POS3/ViT-B, CS naive is **0.4295** versus original naive **0.3949**, difference **+0.0346 [0.0141, 0.0537]**. CS naive versus existing CASR is not distinguishable by this interval: **-0.0039 [-0.0248, 0.0152]**.

The unchanged frozen smart selector transfers to CS: mIoU **0.4648**, gain **+0.0352 [0.0238, 0.0479]**. Frozen fallback gives **0.4684**; its additional IoU change **+0.0036 [-0.0067, 0.0140]** is not established, but harm against CS_MAP decreases from **33.4% to 14.6%** (naive harm 40.6%). Common-protocol best-candidate accuracy rises **47.8% to 54.4%**, oracle mIoU **0.5273**. CS_MAP: PG 0.6690, EPG 0.2706, AP 0.6438, IoU 0.2997. EPG is sensitive to official min-max normalization and should not be treated as calibrated across sources.

Semantic-success proxy declines **82.3% -> 80.9% -> 79.0%** (naive -> smart -> fallback). Smart-minus-naive is -1.4 percentage points [-2.7, -0.1]; fallback-minus-smart -1.9 [-3.7, -0.3]. Do not frame the IoU/harm gains as improved semantics.

All nine methods, paired CIs, harm distributions, candidates, source/checkpoint hashes and 22 representative panels across eight groups are retained. Only one case met the original-map-success/CS-map-failure group. 66 tests passed. Final audit: `outputs/external_baselines/clip_surgery/audit.json`. Main environment and frozen CASR are unchanged; separate demo-only environment isolates OpenCV/NumPy. Raw arrays/checkpoints remain local and excluded from Git. This study and RefCOCOg are not yet pushed.

Defensible contribution: frozen candidate selection and fallback harm reduction transfer to a different attribution source under the common protocol. Do not claim CLIP-to-SAM prompting as novel, overall superiority over official CLIP Surgery, or solved semantic localization. Stop for user review. No automatic RefCOCO extension, Grad-ECLIP, new method, tuning or GitHub push.

## Earlier completed frozen evaluation: RefCOCOg

Read `REFCOCOG_RESULTS.md` and `REFCOCOG_REPRODUCTION.md`, then the full report under `outputs/refcocog_frozen/`. Original REFER UMD test: 2,600 images / 9,602 expressions / 5,023 targets. Standard mIoU: CLIP 0.2173, naive SAM 0.3183, smart 0.3451, CASR final 0.3472. Final-minus-naive +0.0289 [0.0247, 0.0330]. Harm versus CLIP 43.70% -> 30.13%. Correct-instance 63.43%, modest gain over naive (+0.99 percentage points [0.40, 1.57]), without an established gain over CLIP. Long-expression mIoU 0.3293; no-good-candidate rate 58.80%.

Union prior-image overlap 157; strict non-overlap is 2,443 images / 8,894 expressions. Strict final mIoU 0.3492 versus naive 0.3199, paired gain +0.0293 [0.0249, 0.0337]. Strict and standard are nested. CASR is just the internal name for the existing frozen final method: no models, features, thresholds, prompts or geometry changed; no negative points. 61 tests passed. Thirty representative panels and complete CSV/JSON records are saved. RefCOCOg artifacts are local and not yet pushed to GitHub.

The requested referring-expression evaluations are complete. Stop for user review; no additional dataset, external baseline or method changes are authorized by this handoff. Results support discussing external comparisons and paper preparation next, without claiming instance ambiguity is solved. RefCOCOg is now inspected test data and must not be reused for tuning while claiming untouched test evidence.

## Earlier completed frozen evaluation: RefCOCO+

RefCOCO+ is complete: read `REFCOCOPLUS_RESULTS.md` and `REFCOCOPLUS_REPRODUCTION.md`; full tables and 30 panels are in `outputs/refcocoplus_frozen/`. 1,500 images / 10,615 expressions. Final full-image instance mIoU 0.3216, naive SAM 0.3016, smart 0.3224. Final-minus-naive +0.0200 [0.0154, 0.0244]; harm 44.63% to 33.08%. Correct-instance 54.72%, without an established improvement over naive. All 1,500 images overlap prior RefCOCO tests: strict non-overlap is empty and cannot be scored. This is not image-disjoint generalization. Positive-only frozen inference is unchanged. RefCOCOg has subsequently been evaluated as documented above. These reports, code and representative panels are included in this repository update; raw arrays remain local.

## Latest completed development experiment

Read `NEGATIVE_POINTS_RESULTS.md`, `NEGATIVE_POINTS_PLAN.md` and `NEGATIVE_POINTS_REPRODUCTION.md`. RefCOCO VAL negative-point development is complete: 500 images, 3,689 expressions, 1,296 targets. Recommendation **STOP**. All six negative variants lowered mean final mIoU and every variant lowered candidate-oracle mIoU with paired intervals excluding zero. No reliable correct-instance improvement. POS3/final mIoU 0.2950; best negative final row 0.2936. All original frozen components and experiments preserved; no new test predictions and no negative method frozen. Do not retry formulas or refit on these results automatically. New local artifacts are under `outputs/refcoco_negative_points/development/`; 24 panels and all 22 systems are retained.

## Read in order

0. `REFCOCO_RESULTS.md` and `REFCOCO_REPRODUCTION.md`: earlier completed frozen generalization experiment; full tables and 30 panels are in `outputs/refcoco_frozen/`. Reports, metrics, code and representative panels are included in this repository update.
1. `SMART_SAM_RESULTS.md`: current result and its limitations.
2. `SMART_SAM_REPRODUCTION.md`: implementation changes, commands and environment details.
3. `outputs/smart_sam/frozen/FROZEN_METHOD.md` and `model_files/models.json`: frozen method and fitted parameters.
4. `outputs/smart_sam/heldout/HELDOUT_REPORT.md`: complete prospective evaluation.
5. `outputs/smart_sam/development/DEVELOPMENT_REPORT.md`: grouped validation and rejected alternatives.
6. `SAM_DIAGNOSTIC_RESULTS.md`, `COCO_SAM_RESULTS.md`, `REFINEMENT_RESULTS.md`, `REPRODUCTION_NOTES.md`: earlier findings and mathematical ambiguities.

## Current scientific state

RefCOCO UNC testA + testB is now complete: 1,500 images / 10,752 expressions / 3,785 referred instances, zero overlap with either prior cohort. Full-image instance mIoU: CLIP 0.2034, naive SAM 0.2872, smart 0.3081, final 0.3071. Final versus naive gain +0.0199 [0.0155, 0.0244]; harm falls 46.30% to 34.11%. Correct-instance selection is 51.15%, without an established gain over naive. Fallback reduces harm but its IoU change versus smart is -0.0010 [-0.0025, +0.0005]. All frozen components remain unchanged. All 46 tests passed; the final audit verified previous outputs and prediction hashes. RefCOCO is now inspected test data: do not tune on it while claiming untouched test evidence. RefCOCO+ has since been evaluated as described above; RefCOCOg has since been evaluated as documented above.

The original 500 COCO images / 1,000 target pairs are DEVELOPMENT ONLY. They informed diagnostics, feature design, fitting and calibration. Five outer image folds and four inner folds cross-fit the selector for gate training.

A nine-feature standardized linear pairwise logistic ranker selects one of three masks from unchanged SAM ViT-B, prompted by three positive CLIP-attribution points. A log attribution-density threshold of 0.72536122868084263 decides whether to use that mask or retain the original CLIP binary mask. CLIP and SAM weights were not trained. Features, fitted parameters and source hashes were frozen before the second cohort was selected.

The second cohort contains 500 image-disjoint COCO val2017 images / 1,000 pairs, seed 2026. Heldout mIoU: CLIP 0.2699; naive SAM-score selection 0.3949; smart selection 0.4291; frozen smart+density fallback 0.4335. Final versus naive gain +0.0386, paired 95% CI [0.0274, 0.0497]. Harm falls 41.1% -> 26.2%. Semantic success 73.2% -> 73.7% and precise-wrong-object rate 7.8% -> 7.4% do NOT establish significant improvements over naive SAM. Wrong-object rate remains above CLIP's 1.5%. In 49.2% of pairs no candidate reaches IoU 0.5. Combined candidate+fallback oracle: 0.5181.

The heldout cohort has now been evaluated and inspected. Never tune on it and still call that tuning prospective heldout evidence. A future modified method needs a separately defined evaluation plan and untouched test data. No next experiment is authorized by this handoff alone; follow the next user's request.

## What this Git snapshot contains

Code, tests, configurations, all root research notes, structured experiment reports/metrics/manifests, frozen fitted models, hashes, 24 COCO heldout panels and 84 referring-expression study panels. Most older galleries retain their index/manifest but their large image collections are not included. Full ImageNet per-image checkpoint metadata is omitted; aggregate and per-image metric tables are retained.

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

> Read START_HERE.md, REFCOCOG_RESULTS.md, REFCOCOG_REPRODUCTION.md, REFCOCOPLUS_RESULTS.md, REFCOCOPLUS_REPRODUCTION.md, NEGATIVE_POINTS_RESULTS.md, REFCOCO_RESULTS.md, REFCOCO_REPRODUCTION.md, SMART_SAM_RESULTS.md, SMART_SAM_REPRODUCTION.md and the frozen method before doing anything. Preserve completed experiments and negative findings. Explain the current state and identify which raw artifacts are available locally. Wait for my next research instruction; do not retrain, tune on inspected COCO/RefCOCO test data, regenerate existing results automatically, or start external baselines or another dataset without instruction.


Publication update: the user authorized backing up all new studies on 2026-10-01. See BACKUP_STATUS.md and ARTIFACTS.md for the current Git/archive scope; earlier local-only statements describe the time of those studies. No new research run is authorized by this backup task.
