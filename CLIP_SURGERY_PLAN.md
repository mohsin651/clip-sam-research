# Official CLIP Surgery and frozen CASR transfer: pre-inference protocol

Use official xmed-lab/CLIP_Surgery commit d4696d47f49cfe70f49140afe5eb94f94c5f59bc, source unmodified. New outputs only: outputs/external_baselines/clip_surgery. No GT tuning, fitting, category filtering or exclusions. Preserve every earlier output by hashes. Preserve CASR original freeze and fitted ranker/density threshold.

## Official protocol and semantic text

Run the entire unmodified demo.py on its demo.jpg before benchmark; only checkpoint paths and headless plotting redirected. Official CS-ViT-B/16 shares existing OpenAI ViT-B/16 checkpoint; official architecture and feature surgery, full-image RGB warp to 512x512, bilinear map interpolation. For each independent category target use the demo's SINGLE-TEXT branch: default official prompt ensemble and empty-text redundant feature, not label-set-dependent feature subtraction. Category strings are exactly the heldout manifest category fields. CASR's saved prompt (a photo of a CATEGORY) remains untouched and is retained in records. Official internal ensemble wraps the identical semantic category with its 85 templates; do not wrap the already templated CASR sentence again. Text embeddings therefore differ by the official ensemble and this is disclosed, not claimed as verbatim-token parity.

Official CS_OFFICIAL_SAM: similarity_map_to_points on original token similarity, t=0.8, down_sample=2, variable equal positive/negative counts and official coordinate rounding; SAM ViT-H sam_vit_h_4b8939.pth, multimask_output=True, argmax predicted SAM score. Never replace this with positive-three/ViT-B while calling it official. Published full evaluation code is not supplied; claim reproduction of released demo behavior under our protocol, not published numerical results.

## Common protocol and feature compatibility

CS_MAP is official get_similarity_map, min-max normalized by that function, projected bilinearly to full-frame resize geometry then cropped with the existing CLIP crop offsets. No CDA formula/normalization is added. Official map is finite nonnegative, so unchanged CASR features are mathematically defined. Signed unnormalized token similarity is retained for provenance and official Text2Points, but never silently passed to nonnegative energy features.

Common CS_POS3_SAM uses this same crop map and unchanged prompts_from_attribution(k=3,min_distance=32), point inverse geometry, original RGB SAM ViT-B, three candidates and SAM-score argmax. CS_POS3_SMART reuses the same candidates with exact frozen candidate_features, ranker scaler/coefs and feature order. CS_POS3_CASR uses the unchanged smart log-density threshold; rejected masks revert to CS_MAP binary mask. No feature fitting or redefinition. Original official full-frame view/resolution and prompt ensemble differ from CASR's crop/input; this is a controlled downstream-protocol comparison of attribution packages, not an isolated causal comparison of formulas.

## Cohort, metrics and failures

Exactly the existing 500 heldout COCO images / 1,000 pairs, original order and manifest hash. No new subset. Inference reads only existing GT-free manifest, image pixels and frozen config. Twenty pairs technical smoke, reused in full run, no GT scoring until all predictions complete. Save official and common masks, scores, CS maps/tokens, points/labels, features and choices, checkpoint/source hashes. Refuse overwrite and require signature-matched resume.

GT is loaded only in a separate evaluator after full inference and prediction hashes. Reuse category_masks, Geometry.mask_to_crop, mask_metrics, localization_metrics, threshold_for(mean), binary_mask. CS_MAP binary threshold is its per-image mean after existing evaluation min-max convention, matching the CASR binary protocol; it is not an official CS segmentation threshold. PG/AP use official bounded map rankings; EPG uses nonnegative official map energy and is sensitive to its min-max zero point, so is not signed-evidence energy or a calibrated cross-method attribution measure. Do not compute artificial continuous metrics for SAM binary masks.

All original CS map and SAM outputs retained, including poor/constant/empty outputs. If official computation produces nonfinite maps, stop and document without repairing it. No outcomes choose settings. Continuous and binary metrics retain every valid pair. Nine rows: existing CLIP, existing naive POS3, existing final CASR, CS_MAP, CS_OFFICIAL_SAM, CS_POS3_SAM, CS_POS3_SMART, CS_POS3_CASR, existing center control. Reuse existing outputs; recompute own baseline map metrics as needed and check stored binary scores.

Harm is relative to each method's OWN map baseline. Original methods use original CLIP; CS variants use CS_MAP. Report fractions, mean/median/conditional deltas and worst-decile mean. For cross-source harm differences clarify baselines differ. Candidate diagnostics: common POS3 oracle, each selector's chosen IoU, regret and best-candidate accuracy (within 1e-8 of maximum; count tied optima as correct), plus strict argmax identity accuracy for transparency.

2,000 paired whole-image bootstrap resamples, seed 2026, preserving both targets; no multiplicity adjustment. Paired comparisons: official SAM-CS_MAP; CS_POS3_SAM-existing naive; smart-CS_POS3_SAM; CASR-smart; CS_POS3_CASR-existing CASR; CS_OFFICIAL_SAM-existing naive/final (descriptive, SAM model/prompt/view/template confounded). Include IoU and own-map harm contrasts, semantic proxies where applicable. Report both protocols without cherry-picking.

## Examples, records and stop

Three distinct images nearest median CS final IoU per group, stable image/category ties: our map succeeds/CS fails; reverse; both succeed; both fail (IoU >=0.5 success, <0.2 failure); CS naive wrong-category to smart correct with IoU >=0.5; fallback prevents harm; smart lowers IoU; largest absolute official/common IoU differences (explicitly outcome-selected). Groups may overlap. Show original RGB, both maps, GT, existing naive/final, official CS SAM, common CS naive/smart/final, point labels and all candidates with method labels. No prevalence inferred from examples.

Save official repository/license/dependency info, sanity artifacts, manifests, all requested tables/JSON, source signatures, raw arrays and reproduction notes. Explain different SAM checkpoints, geometry, template ensemble and single-text branch throughout. Stop after dense heldout COCO. No RefCOCO, Grad-ECLIP, new methods or automatic push.
