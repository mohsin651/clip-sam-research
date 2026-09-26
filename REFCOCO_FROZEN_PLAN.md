# RefCOCO frozen cross-dataset evaluation

This is evaluation only. Reuse all original CLIP/SAM methods, features, trained selector, density threshold and geometry unchanged; verify `outputs/smart_sam/frozen/integrity.json` before and after inference. Stop on an incompatibility rather than changing the method. Existing experiments remain read-only. No RefCOCO+, RefCOCOg, model fitting or threshold tuning.

## Data and splits

Use original REFER RefCOCO annotations, UNC split, complete testA and testB. Report each split separately and their expression-weighted union. The union is labeled combined test, not a newly invented official split. Download original COCO train2014 images by recorded COCO filename; reuse byte-identical existing COCO images when available. Record sources, hashes, valid/invalid samples and all split counts. No category, target visibility or difficulty filtering. Overlap is exact COCO image-ID overlap against the union of the previous development and heldout 500-image cohorts. Strict non-overlap scores filter those images after predictions without changing the method. Save overlap IDs/counts before inference.

Use each annotation sentence's original `raw` expression verbatim, normal frozen CLIP tokenization, no template or simplification. Record tokenizer preflight. If normal tokenization cannot accept an expression, stop and document rather than truncating or silently excluding it.

## Inference and technical validation

Use all original-resolution RGB images with unchanged CLIP resize/center-crop preprocessing; top-k=3, point minimum separation=32 crop pixels, original coordinate transform, SAM ViT-B multimask outputs. Import existing implementations. New code is a dataset/loop adapter only. Never load GT annotations in inference. Strip the inference manifest to image path, image ID, sentence ID and raw text. IDs only identify artifacts and never enter the selector.

Exactly 20 deterministic samples, sorted image ID then sentence ID, form the technical smoke stage. Validate all dataset IDs/masks/tokenization before inference, but calculate no GT performance until every full-run prediction is complete. Smoke verifies finite maps/scores, points/geometry, model loading, saving/reloading, feature/prediction replay. Full inference reuses those 20 predictions; no repeated predictions or selection based on smoke performance. Reuse per-image SAM embeddings and center candidates. Hash source/config/manifest and checkpoints, retaining resume integrity checks and all numerical failures.

## Metrics and geometry

Primary: per-expression foreground IoU against the specific referred instance over the entire original image. SAM predictions are original-resolution masks; CLIP/fallback masks use the existing inverse crop transform and zero outside the visible crop. This preserves inference but changes the evaluation task from the previous category-union crop benchmark. Report secondary unchanged-crop instance scores separately, including crop-empty targets rather than dropping them. For crop-empty GT, IoU/Dice are defined as zero even for empty prediction, and the count is disclosed; primary valid full-image GT is nonempty.

M0 CLIP, M1 SAM-score, M2 frozen smart selector always-SAM, M3 frozen smart+density fallback, M4 center-point/SAM-score. Dice, precision, recall, pixel accuracy, P@0.5 and P@0.7 accompany mIoU. Harm uses delta tolerance 1e-8. Worst-decile delta is the mean of the lowest ceil(n/10) differences. Compare each method with CLIP and final directly with naive SAM. Do not invent continuous SAM AP/EPG.

Post-inference instance disambiguation: target instance IoU strictly exceeds the largest IoU with any other non-crowd annotation of the same category in the full image. Crowds are not individual competitors; counts are disclosed. With no competitors, max-other IoU=0, requiring positive target overlap; also report the meaningful multiple-instance subset separately. Wrong-instance preference is max-other IoU > target IoU, and ties are separate. Original overlapping masks are retained. Full annotation coverage must be checked before claiming this metric.

Post-hoc expression groups fixed here: whitespace-word length short <=3, medium 4-6, long >=7; spatial terms left/right/front/back/behind/next to/above/below/between/under/over/near/beside; color terms red/blue/green/yellow/black/white/brown/orange/pink/purple/gray/grey; clothing terms shirt/pants/jeans/hat/jacket/dress/shorts/shoes/skirt/coat/sweater/tie; other attributes small/large/big/little/tall/short/young/old/striped/spotted. Regex word boundaries, case-insensitive, original text unchanged. Groups overlap and are descriptive, not verified linguistic labels. Same-category ambiguity one versus multiple full-image instances.

2,000 paired whole-image bootstrap resamples, seed 2026, retaining all expressions per sampled image and expression-macro weighting. Recompute denominators for subgroup resamples. Report mIoU/harm/instance-selection contrasts for naive-CLIP, smart-naive, final-naive, final-CLIP and final-smart. No multiplicity adjustment; disclose this. Oracles and failure proxies are post-hoc only: no candidate reaches IoU 0.5, candidate-choice regret, fallback rejected useful candidate, fallback accepted harmful candidate. These do not causally separate CLIP semantics from SAM proposal errors.

Generate all reports under `outputs/refcoco_frozen/`, save all predictions/candidates, and generate deterministic representative panels after evaluation. Freeze this plan and adapter sources before the 20-sample smoke. Preserve all earlier output hashes. Stop after RefCOCO and report negative outcomes without repair.
