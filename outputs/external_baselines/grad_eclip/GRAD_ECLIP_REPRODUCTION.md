# Grad-ECLIP external baseline and frozen CASR transfer

Official source: https://github.com/Cyang-Zhao/Grad-Eclip, commit `e370e6cb194faf2020f5d1ed268f9d57e91a38e6`, verified against remote HEAD on 2026-10-01. Local checkout `data/Grad-Eclip-source` already existed from earlier source inspection. Tracked files are unchanged. An untracked historical `chefer_segmentation_reference.py` is not part of Grad-ECLIP and is never imported. No license declaration or requirements file was found in the inspected release. Do not infer a license or redistribute the checkout as our code.

## Official code, model and preprocessing

The authors' `grad_eclip_image.ipynb` is the source of executed definitions: cells3,4,9 are compiled verbatim, retaining source filenames/cell identifiers. The adapter supplies their global imports and model; it does not transcribe or replace the attribution equation. Original demo cells5,6,7,10 execute on supplied `dog_and_car.png`, with original four text prompts. Notebook installation commands, inline-plot magic and an unused `open_clip` startup import are omitted; plotting/checkpoint paths are redirected. These are explicit execution accommodations, not method changes. `generate_emap.py` eagerly loads unrelated comparator repositories; it is not used in place of the self-contained official image demo.

OpenAI CLIP ViT-B/16 is officially supported and matches the existing checkpoint. ViT-B/32 also appears in released code. Model checkpoint `data/checkpoints/ViT-B-16.pt` SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`. Native OpenAI CUDA half precision is retained. The official dense forward uses the final layer (`n=1`) with authored single-head/full-channel attention; no heldout layer search. Image gradients are enabled; no inference-mode/no-grad wrapper surrounds image attribution.

Official `imgprocess` resizes the full RGB image bicubically to nearest patch-compatible dimensions with scale_factor=1, applies CLIP mean/std and interpolates positional embeddings. This differs from original CDA's center-crop input and CLIP Surgery's512 warp. Existing saved `a photo of a CATEGORY` strings enter official CLIP tokenization verbatim; no ensemble or prompt optimization. Text embeddings are cached under no_grad in benchmark inference; this does not alter derivatives with respect to the image attention output.

Official `grad_eclip` computes projected Q/K cosine spatial weights, min-max normalization of those weights, gradient-times-value channel contributions, ReLU and the authored layer sum. Native raw nonnegative patch maps are saved. Their own scale is retained; no CDA/CS map or extra phi normalization is used. The authors' torchvision Resize operator maps raw patches into full-frame resized coordinates, then the existing224 center crop is sliced. Native half dtype is retained through interpolation; float32 NumPy storage preserves these values exactly. Binary evaluation alone uses the existing mean threshold; visualization rescales color ranges. EPG is raw nonnegative attribution energy and is not calibrated against other map families.

## Environment and sanity

Benchmark: existing `C:/CREMI/trdp/.venv`, Python3.11.16, torch2.11.0+cu128, NumPy2.4.6. The official plotting demo uses the already isolated `.venv-clip-surgery` overlay for OpenCV4.12.0/NumPy2.2.6 and reads pinned base packages. No package in the main CASR environment is changed. Dependencies: torch, torchvision, OpenAI clip, Pillow, numpy, ftfy, regex, tqdm; cv2/matplotlib only for official demo; existing scipy/pandas/sklearn/SAM/pycocotools for common inference/evaluation. Exact package inventories are in run.json and sanity/sanity.json; the latter records the actual overlay NumPy version explicitly.

Official demo passed model loading, native preprocessing, active gradients, finite/nonconstant/nonnegative maps and dog-versus-car prompt sensitivity. Official figures and raw maps are retained in sanity/. 69 tests passed before benchmark, including executed native-gradient definition, half-precision map crop/scale preservation and unchanged frozen-feature compatibility. Only technical checks on the first20 COCO pairs; no GT scores before full inference. Smoke predictions are reused.

## Shared SAM and CASR

Grad-ECLIP does not provide an official SAM refinement baseline in this inspected release. GE_POS3 variants are explicitly common-protocol experiments. Exact existing `prompts_from_attribution(k=3,min_distance=32)`, inverse point coordinates and SAM ViT-B original RGB preprocessing are reused. Checkpoint `sam_vit_b_01ec64.pth`, SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`; official segment-anything1.0 source commit `dca509fe793f601edb92606367a655c15ac00fdf`. Three candidates per prompt.

The unchanged candidate_features function retains all semantics: nonnegative energy density inside/outside, ratios, coverage, area, SAM score, point containment, agreements, point spread, entropy/concentration. Native Grad-ECLIP maps satisfy that domain. Frozen ranker feature order/scaler/coefficients unchanged; log-density fallback threshold remains0.72536122868084263. Rejected predictions return GE_MAP's binary mask. No fitting, calibration, tuning or feature redefinition. Empty/zero outcomes retain existing conventions and remain in results; nonfinite native maps trigger a documented stop rather than repair.

## Fixed data, evaluation and integrity

Same saved heldout manifest:500 images,1,000 targets. The cohort was already inspected in earlier studies; it is not newly untouched. Annotation-free inference reads images, target strings and frozen parameters. Separate evaluator loads GT only after all predictions complete and hashes verify. Target unions, masks, crop, localization_metrics, mask_metrics and threshold_for/binary_mask match previous evaluation. Original/CS rows and candidates are reused without rerunning models.

2,000 paired whole-image bootstrap resamples, seed2026, preserving both prompts. Harm relative to GE_MAP; cross-source harm references differ. Semantic-success and precise-wrong-object definitions unchanged, overlap proxies rather than manual semantics. Candidate oracles/accuracy/regret use GT post-hoc only, with tied optima tolerance1e-8 and strict-index accuracy also disclosed. All requested comparisons and negative findings remain; no multiplicity adjustment.

Source/protocol/checkpoint signatures are frozen before smoke; prior artifacts are hashed before inference and checked afterward. Resume requires exact signatures and completed runs cannot be overwritten. Official checkout is excluded from Git, as are weights/datasets/raw prediction arrays. A Git clone alone is not a raw-artifact backup.

## Commands (new output destination for reproduction)

From the project directory with restored data and pinned environment:

```
../../.venv-clip-surgery/Scripts/python.exe scripts/sanity_grad_eclip.py
../../.venv/Scripts/python.exe -m pytest -q
../../.venv/Scripts/python.exe scripts/run_grad_eclip_external.py --smoke
../../.venv/Scripts/python.exe scripts/run_grad_eclip_external.py --resume
../../.venv/Scripts/python.exe scripts/evaluate_grad_eclip_external.py
../../.venv/Scripts/python.exe scripts/report_grad_eclip_external.py
../../.venv/Scripts/python.exe scripts/visualize_grad_eclip_external.py
../../.venv/Scripts/python.exe scripts/audit_grad_eclip_external.py
```

Clone the official repository to data/Grad-Eclip-source and check out the exact commit; obtain checkpoints from original sources with matching hashes. Restore the frozen original manifest, prior saved comparison CSVs/candidates/maps and official COCO val2017 data. Paths above are Windows workspace-relative; adjust environment paths on a new machine. Use a separate checkout/output destination; never overwrite this completed study. Protocol and example selection: GRAD_ECLIP_PLAN.md. Stop after this experiment; no automatic new datasets/methods/paper runs or push.
