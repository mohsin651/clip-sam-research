# First frozen SAM extension

Freeze before smoke inference or GT evaluation. No CLIP changes, training, parameter selection on GT, or modifications to baseline outputs.

## Model and inference boundary

Official facebookresearch/segment-anything commit dca509fe793f601edb92606367a655c15ac00fdf, original SAM ViT-B checkpoint sam_vit_b_01ec64.pth. CUDA FP32, evaluation mode, no training. Record checkpoint SHA256. Same saved 1,000 target maps; CLIP is not rerun. All 500 original RGB images are passed to SAM; image embedding reused for both targets and all prompts.

Export a manifest containing only image ID, filename, category ID and prompt text. Inference reads that manifest, image pixels and saved attribution NPZs. The SAM inference module accepts only RGB image, attribution and frozen settings; it never imports COCO or reads annotations, evaluation tables, groups or masks. Ground truth is loaded in a separate evaluation process only after the full 1,000-pair inference completes. Hash every existing baseline file before and after.

## Geometry

Exactly reproduce torchvision/OpenAI CLIP geometry: shorter side 224, longer side integer truncation, then center crop with Python round((resized_dimension-224)/2). For crop point center (x,y), original pixel-center coordinates are ((x+left+0.5)*W/resized_W-0.5, (y+top+0.5)*H/resized_H-0.5), clipped to valid image pixel centers. Box boundaries use pixel-edge coordinates: [xmin+left,ymin+top,xmax_exclusive+left,ymax_exclusive+top] scaled by original/resized dimensions, clamped to [0,W]x[0,H]. SAM's predictor handles its own resize from these original coordinates.

Map attribution back by placing the 224x224 map into a zero canvas of CLIP-resized dimensions then bilinearly resizing to original dimensions. There is no inferred attribution outside the CLIP view. Coverage is computed on this original-image map and original SAM masks. Primary evaluation maps binary SAM masks with nearest-neighbor resize/center crop back to the identical 224x224 baseline grid. Secondary full-original-image evaluation compares SAM to original category unions and a lifted, zero-outside-crop CLIP binary mask; these results must not be mixed with the primary baseline IoU 0.2846.

## Frozen prompts and selectors

- top1: highest raw resized 224x224 attribution, first row-major maximum.
- topk: k=3 positive points, greedy highest remaining strictly positive attribution with minimum Euclidean distance 32 crop pixels (two CLIP patches). Stable row-major ties; fewer points only if positive support cannot supply k. k configurable but fixed to 3 here.
- box: normalize to [0,1], threshold strictly above its mean using the frozen mean rule; 8-connected component containing the global maximum; tight rectangle with exclusive upper edge, no padding.
- point_box: same top1 plus same box.
- center_point: original-image center ((W-1)/2,(H-1)/2), positive; nonsemantic control.
- Ask SAM for three multimask candidates for every strategy. Preserve all masks and scores. No candidate rejection or postprocessing. SAM mask threshold stays the pretrained default (logit>0).
- sam_score: largest predicted IoU score, first-index ties.
- attribution_coverage: largest raw mapped-attribution energy fraction within candidate, first-index ties; no area penalty. Disclose large-mask bias. Center+sam_score is the truly nonsemantic control; center+coverage is also reported but uses attribution for selection.
- Degenerate attribution (if any): center point, full CLIP-view box, flag and retain. No GT-dependent fallback.

## Gates, metrics and analysis

First 25 manifest images (50 pairs) are a technical smoke test. Synthetic landscape/portrait/odd-dimension geometry, actual crop correspondence, point roundtrips, box bounds, finite outputs and saved-mask shapes must pass. Inspect geometry panels. Do not compute smoke GT performance. Freeze signature, then resume remaining images unchanged. No evaluation until all inference completes.

Primary masks: original target-category union within the same CLIP crop; baseline scores/groups exactly preserved. SAM IoU, pixel accuracy, Dice, foreground precision/recall are binary segmentation metrics. Retain continuous CLIP PG/EPG/AP as baseline context only; do not invent SAM continuous attribution metrics.

Semantic success: target-category IoU strictly greater than the maximum IoU with any other annotated category union. Use original other-category masks, without subtracting target overlap, and disclose overlap ambiguity. Also record target pixel overlap versus disjoint distractor pixel overlap and precision. Precise-wrong-object proxy: max other-category IoU>=0.5 AND strictly greater than target IoU. Equality is not success. Category-union GT may penalize a correct single-instance SAM mask when multiple target instances exist.

For each of 10 SAM strategy/selector combinations and both coordinate frames: mean IoU, paired delta versus matching baseline, median delta, fractions improved/unchanged/worsened (absolute tolerance 1e-8), and image-cluster bootstrap 95% intervals. 2,000 resamples, seed 2026. For A/B/C subgroups, resample whole images with sums/counts to preserve pair weighting when an image contributes one or two subgroup targets. Report all 3 groups and all methods. Compare coverage minus score, and top1 minus center, with paired image-bootstrap intervals.

Failure proxies per method: A improved/worsened/unchanged; C improved/worsened/unchanged; B recovered if semantic success and target IoU>=0.5; B precise wrong object if precise-wrong proxy; B otherwise unresolved. These are metric proxies, not verified object identities. Evaluate whether A has the largest gain; do not assume it.

Visuals: at least 50 panels, 10 per prompt strategy, using SAM-score selection as the prechosen display selector. Per strategy select biggest gains, biggest losses, A examples and precise-wrong-object cases; fill deterministically by ID if a category is empty. Deduplicate within strategy. Show all candidates and both selection choices. Add random geometry debug panels from first smoke images, with original/crop/map/lifted-map/point/box. Save exact example manifests.

Report all results, no best-strategy deployment or further tuning. Observed rankings are descriptive, not a new validated choice. Stop after this experiment.
