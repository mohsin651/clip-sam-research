# CLIP Surgery external baseline: reproduction notes

This study evaluates only the existing dense COCO heldout cohort: 500 images and 1,000 category targets. It imports official CLIP Surgery code without modifying it. The benchmark was already inspected in earlier studies; no choices are tuned on its labels in this experiment. This is an external comparison on the fixed cohort, not a new untouched test set.

## Official source and functioning demo

Repository: https://github.com/xmed-lab/clip_surgery. Commit: `d4696d47f49cfe70f49140afe5eb94f94c5f59bc`. Local clone: `data/CLIP-Surgery-source`. No license file or requirements file was present in the inspected commit; no license is inferred. The clone is excluded from Git. The README supplies demos rather than full paper evaluation code and notes differences from paper experiment code. We reproduce released behavior under our evaluation, not the paper's numerical results.

`scripts/sanity_clip_surgery.py` runs the complete original `demo.py` on original `demo.jpg`. Only checkpoint file lookup and headless figure saving are redirected. All 19 figures were saved; maps were finite and nonconstant, and all demo SAM paths completed. Source hashes before and after match. See `outputs/external_baselines/clip_surgery/sanity/sanity.json` and figures.

The demo needs Python, torch, torchvision, NumPy, Pillow, ftfy, regex, tqdm, setuptools/pkg_resources, matplotlib, OpenCV and official segment-anything. It ran in `.venv-clip-surgery`, which reads the existing main environment through a .pth and adds OpenCV-headless 4.12.0.88 and NumPy 2.2.6 locally. Main `.venv` packages were not changed. Benchmark inference uses the original `.venv` with NumPy 2.4.6 and torch 2.11.0+cu128; it needs no OpenCV. Exact package versions are retained in run/sanity JSON.

The benchmark adapter imports the checkout's package under the alias `official_cs` using importlib. Its relative imports and algorithm functions are unchanged. This avoids replacing the installed OpenAI `clip` package used by frozen CASR. No pip installation of a different `clip` package into the main environment is needed. Clone the official repository into `data/CLIP-Surgery-source` and check out the commit above before invoking the adapter.

## Models and official inference

- Official `CS-ViT-B/16` with the existing original OpenAI `ViT-B-16.pt`, SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`. Official architecture surgery and feature surgery are imported directly. Model runs float32.
- RGB full-image warp to 512x512 with bicubic interpolation and official CLIP mean/std. This differs from original CASR's center crop.
- The official single-target branch is used independently for each saved category string. Default 85-template official text ensemble and the empty-string ensemble redundant feature are retained. The original manifest category and CASR prompt are both saved. Official internal templates therefore differ from the original single `a photo of a CATEGORY` sentence; semantic targets are identical, embeddings are not.
- `clip_feature_surgery(..., redundant_feats=empty_ensemble)` produces signed token similarities. `get_similarity_map` min-max normalizes tokens then interpolates bilinearly. No CDA attribution or normalization is added.
- Official Text2Points uses the unnormalized tokens, `t=0.8`, `down_sample=2`, equal variable positive and negative counts, official ranking, integer rounding and original-image coordinates.
- Official SAM ViT-H: `sam_vit_h_4b8939.pth`. The official path uses three candidate masks and argmax SAM predicted score, original RGB and SAM preprocessing.

Verified ViT-H SHA256: `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` (2,564,550,879 bytes). Acquisition URL and verification are in `sam_h_acquisition.json`. Common POS3 uses unchanged ViT-B SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.

Both SAM paths use the already pinned official `segment-anything` package 1.0, source commit `dca509fe793f601edb92606367a655c15ac00fdf`, as specified by `requirements-sam.txt`.

## Common protocol and frozen transfer

The official map is interpolated into the existing full-frame resized geometry and then sliced by the unchanged 224 center-crop offsets. All metrics use the same target category unions and crop as original heldout results. Binary masks use the existing mean-threshold rule. EPG is computed on the official nonnegative map and depends on its min-max zero point; it is not calibrated signed evidence.

The common variants use exactly `prompts_from_attribution(k=3,min_distance=32)`, original point inverse geometry, SAM ViT-B and three candidates. `candidate_features`, feature order, scaler, ranker coefficients and density fallback threshold are imported unchanged. Rejected SAM predictions revert to the CS binary map. Official nonnegative maps satisfy the feature domain; no incompatible signed feature is substituted.

CS_POS3_SAM selects SAM score; CS_POS3_SMART selects the frozen ranker; CS_POS3_CASR adds frozen density fallback. These are controlled hybrids, not the official external pipeline. The downstream protocol is shared with original CASR, while upstream image view, resolution and text ensemble differ. This comparison cannot isolate attribution formula alone. Official ViT-H results additionally differ in SAM backbone and positive/negative prompting.

## Integrity and execution

66 tests passed before inference. The first 20 pairs receive technical checks only and are reused. Inference imports no COCO annotation reader or evaluation functions. GT evaluation runs separately after all 1,000 predictions complete and hashes verify. Prior experiment files, original freeze, image files, checkpoint hashes, adapters and protocol are protected by recorded hashes. Resume requires identical signatures. Completed runs cannot be overwritten.

From the project directory, use the existing pinned Python:

```
python -m pytest -q
python scripts/run_clip_surgery_external.py --smoke
python scripts/run_clip_surgery_external.py --resume
python scripts/evaluate_clip_surgery_external.py
python scripts/report_clip_surgery_external.py
python scripts/visualize_clip_surgery_external.py
python scripts/audit_clip_surgery_external.py
```

The official sanity prerequisite, in a fresh destination, is `../../.venv-clip-surgery/Scripts/python.exe scripts/sanity_clip_surgery.py`. The demo-only environment was created with uv using the existing Python, a `.pth` pointing to the original `.venv/Lib/site-packages`, and local `opencv-python-headless==4.12.0.88` plus `numpy==2.2.6` installed without dependencies. This reuses pinned torch/SAM while isolating the OpenCV NumPy requirement. Recreate the paths for the new machine; do not point a shared environment at this overlay. The benchmark uses `../../.venv/Scripts/python.exe`, not the demo overlay. `scripts/download_cs_sam.py` retrieves the official ViT-H checkpoint and saves its acquisition record.

For a fresh reproduction, clone the exact official commit, obtain verified checkpoints, restore the original heldout manifest/frozen parameters and COCO images/annotations, and use a separately named output directory in a separate checkout. Large datasets, checkpoints and raw arrays are excluded from Git. These commands intentionally refuse to overwrite this completed study.

Statistics: 2,000 paired whole-image bootstrap resamples, seed 2026, both targets together. Harm uses each source's own map baseline. Candidate oracles are post-hoc diagnostics only. All outcomes, failures and fallbacks remain; no fitting, tuning, category exclusions or formula search. Representative examples are outcome-selected by predeclared rules; they do not estimate prevalence. No RefCOCO or other external baseline follows automatically.
