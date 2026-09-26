# First frozen CLIP-attribution + SAM findings

Completed 2026-09-23 on the existing 500 images / 1,000 target pairs. No CLIP changes, training or GT-based prompting/candidate selection. The 50-pair smoke gate checked only execution and geometry. All baseline files remain unchanged.

Official SAM ViT-B: facebookresearch/segment-anything commit dca509fe793f601edb92606367a655c15ac00fdf, package 1.0. Checkpoint sam_vit_b_01ec64.pth, SHA256 ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912. Original-image SAM inference, peak allocated CUDA memory 2.73 GiB.

## Primary mIoU: identical CLIP crop evaluation

| Prompt | SAM-score selection | Attribution-coverage selection |
|---|---:|---:|
| Top-1 point | 0.3004 | 0.3865 |
| Top-3 points | 0.4005 | 0.3947 |
| Attribution box | 0.3273 | 0.3512 |
| Point + box | 0.3645 | 0.3742 |
| Center control | 0.1048 | 0.1723 |

Frozen CLIP binary baseline: **0.2846**. Center+SAM-score is fully nonsemantic; center+coverage uses CLIP at candidate selection.

Highest observed overall: top-3 + SAM score, mIoU **0.4005**, delta **+0.1158**, image-cluster bootstrap 95% CI **[+0.0983, +0.1339]**. Median delta +0.0466; 56.0% improved, 44.0% worsened, 0 unchanged. Ranking is descriptive, not a statistically established winner over every alternative.

## Frozen subgroup results for top-3 + SAM score

| Group | Pairs | Baseline IoU | SAM IoU | Delta | Delta 95% CI |
|---|---:|---:|---:|---:|---|
| A: correct/coarse proxy | 398 | 0.3111 | 0.4925 | +0.1814 | [+0.1509, +0.2126] |
| B: wrong/unconfirmed | 455 | 0.1567 | 0.2369 | +0.0802 | [+0.0559, +0.1051] |
| C: already good | 147 | 0.6091 | 0.6577 | +0.0486 | [+0.0069, +0.0894] |

Group A has the largest gain. Its highest observed configuration is top-1 + coverage: IoU **0.5362**, delta **+0.2251**, CI [+0.1953, +0.2548].

## Answers and limitations

- **Does guidance matter?** Yes: top-1 + SAM score exceeds center + SAM score by +0.1956 IoU, paired CI [+0.1712, +0.2201].
- **Does coverage beat SAM score?** Not uniformly. It improves top-1 by +0.0861 IoU, box by +0.0239 and point+box by +0.0097; top-3 changes -0.0058 with an interval including zero. Top-1 coverage nevertheless lowers semantic-success rate by 4.6 percentage points relative to SAM score.
- **Wrong objects?** Top-3 + SAM score yields 71/1,000 precise-wrong-object proxy cases (7.1%): another category's union IoU is >=0.5 and exceeds target IoU. The rule is not manual semantic verification.
- **Semantics versus boundaries?** Top-3 + SAM-score semantic success is 75.8% versus baseline 68.5%, where success means target IoU exceeds every other category IoU. It recovers 87/455 B cases to target IoU>=0.5 plus semantic success. The largest benefit is in A; SAM receives no text and does not independently solve semantics.
- **Further research justified?** Yes, for controlled development. The gain and nonsemantic control support attribution-guided refinement, but part-only masks, wrong-object masks, overlapping COCO annotations and category-union versus single-instance mismatch remain important. No further tuning was performed.

Original-image secondary evaluation also improves: matching lifted CLIP baseline 0.2609 -> top-3 + SAM score 0.3884. Do not mix these with crop-matched values above.

All 15,000 candidates and both selectors are preserved; 22,000 method/frame result rows and 50 distinct visual example cases saved. 34 tests passed. A floating-point box-edge clamp corrected the implementation to match the prewritten plan before GT evaluation; the failed technical attempt is archived and documented.

- [Full report and runtime](outputs/coco_sam/RESULTS.md)
- [Frozen protocol](COCO_SAM_PLAN.md)
- [All 50 example panels](outputs/coco_sam/examples.html)
- [Representative visual review](outputs/coco_sam/VISUAL_REVIEW.md)
- [Technical correction history](outputs/coco_sam/TECHNICAL_NOTES.md)
- [Subgroup results](outputs/coco_sam/subgroup_results.csv)
- [Paired comparisons](outputs/coco_sam/paired_comparisons.json)

Official source: [Segment Anything repository](https://github.com/facebookresearch/segment-anything).
