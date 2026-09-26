# Research continuity

Read `START_HERE.md`, `SMART_SAM_RESULTS.md` and `SMART_SAM_REPRODUCTION.md` before changing the experiment. Follow the current user's task; do not automatically start a new research run.

Preserve completed results and their original provenance. The original 500 COCO images are development data. The second 500-image cohort was evaluated after method freeze and is now inspected; do not tune on it while describing the result as untouched heldout evidence.

Keep inference-time features separate from GT evaluation. Do not use masks, GT boxes, GT areas, IoUs or diagnostic groups as model inputs. Report negative findings and all retained failures. New methods should use new output destinations and explicit protocols instead of overwriting historical artifacts or changing their integrity manifests.

The frozen source/checkpoint hashes are in `outputs/smart_sam/frozen/integrity.json`. Preserve the frozen implementation; use a separately named extension for new method development. Git excludes large raw artifacts; consult `ARTIFACTS.md` before assuming missing files are errors or rerunning a completed study.

Use the existing scripts, environments and saved metrics where available. Tests are under `tests/`; the last completed suite had 41 passing tests. Avoid rerunning model inference when the task can be completed from saved CSV/JSON results.
