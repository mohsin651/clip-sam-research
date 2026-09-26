# Representative visual inspection

These are three automatically selected cases from the predeclared SAM-score gallery. Inspection does not alter any metric, prompt, candidate choice or parameter.

- **Bottle / image 000000570782 / top-3:** crop IoU 0.1039 -> 0.9245. The CLIP map reaches the bottle but its thresholded mask also includes nearby desk objects. SAM's confidence-selected candidate closely follows the bottle. This is a clear example consistent with the intended boundary-refinement benefit.
- **Motorcycle / image 000000256192 / top-3:** crop IoU 0.6882 -> 0.0761. All three points are around the rear/seat area. The highest-confidence candidate selects only a motorcycle part; the coverage-selected candidate includes almost the whole vehicle. This illustrates part-versus-whole ambiguity and why one selector need not be best in every case. The experiment does not switch selectors using GT.
- **Person / image 000000410880 / top-3:** crop IoU 0.0156 -> 0.0000. CLIP focuses on a large teddy bear, and SAM precisely segments that bear while the annotated person is elsewhere. Sharp segmentation cannot repair this wrong semantic prompt location.

Panels:

- [Bottle improvement](examples/sam_topk_points_000000570782_44.png)
- [Motorcycle part-only failure](examples/sam_topk_points_000000256192_4.png)
- [Teddy bear instead of person](examples/sam_topk_points_000000410880_1.png)

These illustrate mechanisms, not their prevalence. Prevalence is measured by the full 1,000-pair tables and the explicitly labeled metric proxies in RESULTS.md.
