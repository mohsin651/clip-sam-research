# Frozen smart SAM method

Frozen UTC: 2026-09-26T12:20:19Z. No heldout cohort exists at freeze time.

Selected method: unchanged refined CLIP attribution -> three positive points -> unchanged SAM ViT-B three candidates -> nine-feature linear pairwise logistic ranker -> density fallback. Use selected SAM iff log((density_inside+1e-12)/(density_outside+1e-12)) >= 0.72536122868084263; otherwise retain the exact original CLIP binary mask. Argmax ties use first candidate. Candidate features are measured on original-resolution masks, attribution zero outside the visible crop. Primary scoring projects masks into the original CLIP crop.

Why chosen: five-fold image-grouped development mIoU: SAM score 0.400482; smart selector 0.441898; smart+density fallback 0.446508; smart+logistic fallback 0.438331. The simple threshold gives the highest observed out-of-fold mIoU. Logistic classification accuracy did not translate into a better segmentation decision, and its results remain disclosed. No search beyond the predeclared models and nine density deciles plus always/never. Development is exploratory, not final evidence.

Ranker is fitted to all development candidate pairs after validation. Fallback threshold is calibrated using out-of-fold smart-selected masks, never in-sample fitted selector outcomes. Inner four-fold cross-fitting trains gates within each outer development fold. Final comparison logistic gates use all outer out-of-fold choices. Candidate rows and both prompts always remain grouped by image. Final model files contain feature order, means, scales, coefficients, intercepts and thresholds in human-readable JSON.

All 18 extracted feature definitions and nine ranker / 14 logistic-gate features are in frozen_config.yaml and hashed scripts/smart_sam_core.py. Pair-level context cancels in an additive within-pair ranker, so it is reserved for fallback. Raw density, ratio and duplicate point counts remain diagnostic columns rather than redundant learned predictors. No GT-derived or category-ID features are used.

Heldout seed 2026; 500 fresh COCO val2017 images excluding all 500 development IDs; same minimum 502 crop pixels and 25% retained area, two eligible targets each. Freeze precedes cohort construction. All inference outputs saved before any GT evaluation. Frozen smart+density is the sole final proposed method; learned logistic controls remain predeclared comparisons. No method changes based on heldout results.

Checkpoints: CLIP ViT-B/16 SHA256 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f; SAM ViT-B SHA256 ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912. No CLIP/SAM weights are trained. Full source, checkpoint, model and config hashes: integrity.json. Protocol: ../../../SMART_SAM_PLAN.md.
