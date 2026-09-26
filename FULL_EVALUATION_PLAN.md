# Full validation evaluation

Evaluate all 12,419 official ImageNet-S919 validation IDs using the frozen
`config_refined.yaml` method from the 500-image study. No new selection or tuning.
Keep the baseline, G_cls, refined full, normalized-prior-only and literal-full
controls. Report the original target-class mask and all-labeled-foreground mask
separately, both with the frozen mean-score threshold.

Preserve all 500-image files and their dataset provenance. Extract the same
pinned mirror shards already downloaded; verify every mask against the official
archive, every class against the official source mapping, and every image/mask
dimension. Full-data provenance is written separately as `provenance_all.json`.

Use the same FP32 deterministic forward, score, gradient, raw Q/K coordinates,
normalization, masks, crop and metric definitions. Skip computing unused projected
Q/K diagnostic alternatives; this does not change any reported map. Verify the
first 20 new maps against the saved refined smoke maps before proceeding.

Save raw 14x14 maps for every image and all five variants, all per-image metrics,
source/configuration hashes and confidence intervals. Save visual panels for the
fixed first 20 only. Append metrics in complete per-image checkpoint files so a
restart never silently skips images. Degenerate maps are flagged and retained
with the established zero-map metric conventions; nonfinite outputs stop the
run. An image with no valid annotation pixels is an explicit failure, never
silently dropped. Report all diagnostics and actual denominators.

Report the full 12,419, the prior 500 and the additional 11,919 separately.
The additional cohort was not used to select the refinement. Bootstrap intervals
describe image variability; evaluating all validation images eliminates subset
sampling uncertainty for this finite validation set, not uncertainty about
generalization to other data.

This remains the documented CDA-inspired refinement, not a claim of exact
reproduction of ambiguous published equations or evaluation protocols.
