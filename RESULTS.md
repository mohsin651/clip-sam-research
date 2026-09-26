# Small reproduction study

Completed 500 fixed ImageNet-S919 validation images, all three variants, seed 42.
Device: NVIDIA RTX 4000 Ada Generation. Checkpoint: OpenAI CLIP ViT-B/16. Attention mode: full.

## Measured results

| Variant | PG | EPG | PAcc | AP | mIoU |
|---|---:|---:|---:|---:|---:|
| baseline | 0.7320 | 0.4668 | 0.7857 | 0.6531 | 0.3780 |
| gcls | 0.7340 | 0.4668 | 0.7857 | 0.6531 | 0.3780 |
| full | 0.1460 | 0.1463 | 0.6909 | 0.2742 | 0.0059 |

## Paired uncertainty

All intervals are image-level percentile bootstrap 95% intervals (2,000 resamples, seed 2026).

| Comparison | Metric | Mean difference | 95% interval |
|---|---|---:|---|
| full_minus_baseline | pg | -0.58600 | [-0.63200, -0.54000] |
| full_minus_baseline | epg | -0.32052 | [-0.34425, -0.29682] |
| full_minus_baseline | pacc | -0.09485 | [-0.10945, -0.07899] |
| full_minus_baseline | ap | -0.37898 | [-0.40193, -0.35608] |
| full_minus_baseline | iou | -0.37205 | [-0.39355, -0.35006] |
| full_minus_gcls | pg | -0.58800 | [-0.63400, -0.54395] |
| full_minus_gcls | epg | -0.32052 | [-0.34425, -0.29682] |
| full_minus_gcls | pacc | -0.09485 | [-0.10945, -0.07899] |
| full_minus_gcls | ap | -0.37898 | [-0.40193, -0.35608] |
| full_minus_gcls | iou | -0.37205 | [-0.39355, -0.35006] |

## Interpretation

This study tests the explicitly documented full-channel interpretation. It does not establish exact reproduction of the paper.
Baseline and Grad-Only are theoretically proportional in this single-head graph; their scale-invariant metrics cannot reproduce the distinct Table 5 rows. This mathematical limitation is unit-tested.
Their measured PG differs on one of 500 images: FP32 interpolation moves an almost-tied edge maximum across the GT boundary. The max-normalized maps differ by only 4.2e-7 on that sample (ILSVRC2012_val_00014552). This is not evidence of a channel-weighting improvement.
Full spatial weighting improves the measured mean of: none.
Full spatial weighting lowers the measured mean of: pg, epg, pacc, ap, iou.
Use the paired intervals above to assess uncertainty. No parameters were adjusted to match published values.
The printed Eq. (11) collapses space to a scalar; our patchwise interpretation and CLS inclusion are documented in REPRODUCTION_NOTES.md. Author clarification is required to settle these discrepancies.

## Validation and limitations

Zero maps: 0; constant maps: 0; failed rows: 0.
Absent original target masks: 57; absent target after crop: 58. These samples were retained.
Total experiment wall time: 335.9 seconds; average shared attribution time: 79.9 ms/image.
The prior 20-image smoke run checked all variants and three correct-versus-different-prompt map pairs.
Mirror masks were checked pixel-for-pixel against the official archive; all original image classes were checked against the official source mapping. Original image bytes are obtained from an unofficial mirror.
Evaluation uses the CLIP crop, target-class binary masks, void exclusion and a configurable 0.5 min-max threshold. These protocol choices are not fully specified by the paper.

See outputs/experiment_500/summary.json for per-variant SD and confidence intervals, and run.json for complete provenance.
Launch the saved-results dashboard with `python scripts/run_dashboard.py`.
