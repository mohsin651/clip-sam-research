# Technical correction before evaluation

The initial 50-pair smoke test passed. Full inference stopped at image 000000079651 because a mapped box boundary was 478 + 5.68434189e-14 instead of exactly 478. This triggered the strict image-bound assertion before GT evaluation.

The frozen plan already specified clamping original-image box coordinates to valid image edges. The initial implementation omitted this final clamp. Added np.clip to implement the stated rule exactly. No prompt-selection threshold, k, candidate selector or attribution parameter changed. No COCO GT performance was calculated or used.

The initial checkpoints and run metadata are preserved in technical_box_roundoff_archive. The 50-pair technical gate and full inference were restarted under the corrected source signature. A separate regression test covers the 640x478 image geometry.
