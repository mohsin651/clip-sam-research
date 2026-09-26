# Frozen refinement protocol

The initial literal interpretation and its 500-image results remain unchanged.
The user authorized source-informed refinements after seeing those results.
This is exploratory method development, not an exact paper reproduction.

## Evidence and choices, fixed before the new 500-image run

- Grad-ECLIP's official `generate_emap.py` (commit
  e370e6cb194faf2020f5d1ed268f9d57e91a38e6) normalizes cosine spatial weights
  with per-image min-max scaling. It also uses single-head final attention.
  https://github.com/Cyang-Zhao/Grad-Eclip/blob/e370e6cb194faf2020f5d1ed268f9d57e91a38e6/generate_emap.py
- Its code uses output-projected Q/K. We tested that as a diagnostic, but keep
  the original Q/K coordinates for CDA to avoid changing the meaning of G_Q.
- The primary correction is `phi01=(phi-min(phi))/(max(phi)-min(phi))`, then
  `W=phi01+ReLU(cos(G_Q,K))`. No fitted coefficient, model training, mask input
  to attribution, image selection, or paper-score matching objective is used.
  This explicitly changes printed Eq. (7)/(10); it is a CDA-inspired refinement.
- Preserve full-channel attention. The initial hypothesis that native attention
  would fix the problem was rejected by the 20-image diagnostic and the
  predecessor implementation. All diagnostic alternatives are saved, including
  native results, raw weights, min-max weights and projected-coordinate weights.
- For mask metrics use the per-image mean score as threshold, following the
  cited Transformer Explainability implementation's threshold rule:
  https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/imagenet_seg_eval.py
  We retain this project's foreground-only per-image IoU definition. That
  reference also has other metric/resize choices which we do not claim to copy.
- Keep the original target-class foreground mask as the primary protocol.
  Also report all-labeled-foreground masks as a clearly labeled secondary
  protocol, and report fixed 0.5 thresholds for both. Scores across different
  mask/threshold protocols must not be presented as method-only improvements.
- Retain seed-42 IDs, checkpoint, prompt, aligned crop, ignored pixels and all
  samples. No hyperparameter is optimized against the 500-image paper values.

## Development and evaluation

The existing 20 smoke IDs were used to diagnose/select these choices.
`outputs/refinement_diagnostics/protocol.json` describes the initial candidates;
its initial native-attention hypothesis was rejected as explained above.
The remaining 480 IDs are not used to select among these new alternatives.
However, their original implementation results were already seen, so the 480
are an internal remaining-sample check, NOT a pristine independent test set.
Report all 500, development 20 and remaining 480 separately.

Run the baseline, G_cls and refined full variant on the same 500. Retain a
minmax-prior-only control to test whether the semantic R term helps; do not
claim class disentanglement gains if that control is as good or better.
Compare the old literal maps under the same new mask/threshold protocols, using
the saved raw maps, to isolate the normalization change from evaluation changes.
Use fixed bootstrap uncertainty and paired per-image differences.

The single-head scalar-gradient identity remains: this correction cannot
honestly recreate distinct baseline/G_cls rows apart from numerical ties.
