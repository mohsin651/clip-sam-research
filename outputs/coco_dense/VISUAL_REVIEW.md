# Small representative visual review

This is an inspection of the three automatically selected median-IoU representatives, not manual classification of the full experiment. It does not alter scores, selection, thresholds or diagnostic groups.

- **Dining table, 000000575970 / category 67 (group A):** the attribution reaches the table but the binary mask also covers chairs and parts of the room. PG=1 and target ratio=0.6233 coexist with IoU=0.3284. This illustrates coarse localization with substantial spill, not proof that boundary refinement alone will repair it.
- **Hot dog, 000000513567 / category 58 (group B):** attribution visibly concentrates around both hot dogs, yet the broad map extends onto people. PG=1, IoU=0.1167 and target ratio=0.2917. This is a concrete reason to call B “wrong OR unconfirmed target localization,” rather than claiming all B cases choose the wrong object. The aggregate-energy rule penalizes small objects surrounded by large annotated distractors.
- **Cat, 000000416330 / category 17 (group C):** the map concentrates on the cat's face and covers much of its region, with PG=1, IoU=0.5993 and target ratio=0.8748. The bed/cat prompt comparison changes substantially. The bed annotation covers nearly the whole crop, including cat pixels, illustrating why category-mask overlap and target-priority handling must remain disclosed.

All three panels and their paired-prompt comparisons are linked in RESULTS.md. The full random/best/worst/confusion gallery is examples.html. None of these three examples was selected manually for a favorable result.
