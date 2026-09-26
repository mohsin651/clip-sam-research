# Smart candidate selection and optional CLIP fallback

This extension reuses the frozen refined CLIP attribution and official SAM ViT-B, including their existing checkpoints, geometry and three-point candidate strategy. It changes only candidate selection and the optional decision to keep the original CLIP binary mask. It does not train CLIP or SAM.

The old 500-image cohort is explicitly development data. All 3,000 saved top-three SAM candidates are reused. Features and development GT labels are separate CSVs. The nine-feature pairwise linear logistic ranker and 14-feature comparison logistic gates use standardized features. Five outer image folds keep both targets and all candidates together. Four inner image folds provide cross-fitted candidate choices for gate training inside each outer fold. Final models are fitted only after out-of-fold evaluation.

The development winner is the smart ranker plus a simple density threshold, not the logistic gate. The latter reduces harm more but sacrifices mIoU; those results remain in the report. The final log-density threshold is 0.72536122868084263. It is calibrated on out-of-fold selected development masks. Exact parameters, feature order, scaler values and coefficients are saved as JSON. The threshold is applied before any heldout evaluation.

The freeze is at `outputs/smart_sam/frozen/`: config, explanatory notes, model parameters and SHA256 integrity manifest. It precedes cohort selection. Heldout sampling uses seed 2026 on eligible remaining COCO val2017 images, excludes all old 500 IDs, and retains exactly two targets per image. Annotation access for eligibility occurs in a separate selection script. Inference receives a stripped manifest containing image filename and target text/category ID, with no masks, boxes, areas, groups or outcomes. Category IDs identify saved artifacts and never enter model features.

Reproduction order in a clean output destination:

```powershell
python scripts/develop_smart_sam.py
python scripts/report_smart_sam.py development
python -m pytest -q
python scripts/freeze_smart_sam.py
python scripts/select_smart_sam_heldout.py
python scripts/run_smart_sam_heldout.py
python scripts/evaluate_smart_sam.py
python scripts/report_smart_sam.py heldout
python scripts/visualize_smart_sam.py
python scripts/audit_smart_sam.py
```

Freeze, cohort selection and heldout inference refuse to overwrite their existing artifacts. Do not delete existing study outputs to rerun. Reporting and audit can be repeated without inference or method changes. Fitting and extraction artifacts are development-only. The old COCO baseline, first SAM experiment and diagnostic outputs are protected by a before/after hash audit.

Primary evaluation uses the unchanged 224-pixel CLIP crop and per-image mean threshold. Confidence intervals use 2,000 paired whole-image bootstrap resamples, seed 2026. Both targets stay together; subgroup resamples preserve target-pair weighting. Worst-decile harm is the mean of the lowest 10% of deltas, not their quantile boundary. Semantic success and precise-wrong-object measures retain the original category-union definitions and limitations. A rejected SAM mask is an unchanged CLIP result, not an excluded sample.

Environment repair: the existing uv Python launcher required restoring its missing `cpython-3.11-windows-x86_64-none` junction to the installed `cpython-3.11.16-windows-x86_64-none` directory. No packages, model weights or previous experiment files were changed. Full environment details are saved in heldout/run.json. Initial Windows DLL loading took longer than subsequent imports.

Study protocol: [SMART_SAM_PLAN.md](SMART_SAM_PLAN.md). Development: [DEVELOPMENT_REPORT.md](outputs/smart_sam/development/DEVELOPMENT_REPORT.md). Frozen method: [FROZEN_METHOD.md](outputs/smart_sam/frozen/FROZEN_METHOD.md). Heldout: [HELDOUT_REPORT.md](outputs/smart_sam/heldout/HELDOUT_REPORT.md). Examples: [examples.html](outputs/smart_sam/heldout/examples.html).
