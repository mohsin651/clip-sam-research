# CDA-CLIP localization reproduction

The newer source-informed refinement is described in [REFINEMENT_PLAN.md](REFINEMENT_PLAN.md)
and [REFINEMENT_RESULTS.md](REFINEMENT_RESULTS.md). Run it with
`python scripts/run_refinement.py --num-samples 20`, then `--num-samples 500`.
Inspect it with `python scripts/run_refined_dashboard.py` at http://127.0.0.1:8052.
The original literal
study below remains intact for comparison. The refinement changes spatial
normalization and mask thresholding; it is not claimed as an exact reproduction.

Controlled, inference-only study using OpenAI CLIP ViT-B/16 on a deterministic
ImageNet-S919 validation subset. Read [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)
before interpreting results: the supplied paper has scalar/map and attention
ambiguities that materially affect Table 5 reproduction.

## Environment

The working environment is `C:\CREMI\trdp\.venv`, Python 3.11, CUDA-enabled
PyTorch, RTX 4000 Ada 20 GB. No system Python was modified.

From this directory in PowerShell:

```powershell
& C:/CREMI/trdp/.venv/Scripts/Activate.ps1
python scripts/verify_model.py
python -m pytest -q
python scripts/download_data.py --num-samples 500
python scripts/verify_dataset.py
python scripts/smoke_test.py --num-samples 20
python scripts/run_localization.py --num-samples 500
python scripts/audit_outputs.py
python scripts/write_report.py
python scripts/run_dashboard.py
```

Dashboard: http://127.0.0.1:8050 . It reads saved files without loading CLIP.
To inspect the smoke run: `python scripts/run_dashboard.py --output outputs/smoke_test`.

On a new machine create `.venv` using Python 3.11, install pinned requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

`environment.txt` records the actual validated machine and package versions.
The CUDA wheel pins are platform-specific; do not use CPU wheels for this study.

## Runs

`config.yaml` controls prompt, threshold, attention mode, seeds, paths and
bootstrap count. Paths in scripts resolve relative to this project.
`--variant baseline|gcls|full|all` selects variants; default all shares one
forward/backward and the exact same saved subset IDs. `--num-samples all`
uses all official validation IDs after downloading/extracting them.

The 500 run requires a successful 20-image smoke run with matching source
hashes and configuration. Existing results are protected from overwriting;
use a fresh `--output` directory for another run. For a separately labeled
native-attention sensitivity study, change `attention_mode` in config and
run its smoke test before its larger study. Never combine attention modes
in one results table without labeling them.

## Artifacts

- `outputs/smoke_test`: 20-image maps, panels, diagnostics and prompt comparison.
- `RESULTS.md`: measured study results, paired intervals and a candid reproduction assessment.
- `outputs/experiment_500/subset_ids.txt`: fixed seed-42 image IDs.
- `per_image_results.csv`: all metrics, cosine scores, map flags and shared timing.
- `summary.csv` / `summary.json`: means, sample SD, bootstrap 95% CIs and paired differences.
- `run.json`: environment, checkpoint/hash, source hashes, configuration, dataset provenance,
  subset hash, status and total runtime.
- `images/<id>`: 14x14 raw and normalized maps, resized scores, binary masks,
  overlays and panels for all variants; first three also contain tensors and diagnostics.
- `data/ImageNetS919/provenance.json`: pinned mirror revision, shard and selected-file hashes.

Quantitative EPG uses raw energy. Mask metrics use the configurable normalized
threshold; AP/PG use continuous raw scores. Evaluation is on the CLIP crop with
aligned masks and ignored-label exclusion. Figures label that crop explicitly.

Unit tests cover hand-computed metrics/equations, attention forward and gradient
equivalence, the single-head gradient identity, deterministic subsets and spatial
mask alignment. The model verification additionally checks the real pretrained
checkpoint on CUDA. Neither synthetic test metrics nor paper values are reported
as experiment measurements.

## Frozen refinement and complete validation

The implementation changes and selection history are recorded in
`REFINEMENT_PLAN.md`; the original 500-image findings are in
`REFINEMENT_RESULTS.md`. The complete 12,419-image procedure is frozen in
`FULL_EVALUATION_PLAN.md`, using `config_refined.yaml` without retuning.

```powershell
python scripts/prepare_full_dataset.py
python scripts/run_full_evaluation.py --resume
python scripts/report_full_evaluation.py
python scripts/run_full_dashboard.py
```

Dataset preparation uses the cached pinned mirror shards and official mask
archive. The evaluation saves resumable per-image checkpoints to
`outputs/refined_all/checkpoints`. Reporting audits all maps and recorded
source hashes before writing `FULL_RESULTS.md`, protocol summaries and paired
bootstrap intervals. The full dashboard uses port 8053 and saved maps only.
The prior 500-image artifacts and dashboards remain available separately.

## COCO multi-object experiment

`COCO_DENSE_PLAN.md` freezes selection and evaluation rules for the same refined
attribution on official COCO 2017 validation. Install `requirements-coco.txt`
into the existing environment, then run:

```powershell
python scripts/prepare_coco.py
python scripts/select_coco_dense.py
python scripts/run_coco_dense.py
python scripts/report_coco_dense.py
```

Selection is deterministic and refuses to overwrite its manifest. Use
`--resume` for interrupted inference. Results, image-cluster bootstrap intervals,
target/distractor and prompt-switch diagnostics are saved in
`outputs/coco_dense/RESULTS.md`. Open `outputs/coco_dense/examples.html` for
the visual examples. All 500 prompt-switch panels and all raw maps are saved.
This experiment adds no SAM and changes no frozen attribution source.

## First SAM extension

The separate `COCO_SAM_PLAN.md` freezes SAM ViT-B prompting and candidate
selection on the same 1,000 COCO pairs. Existing baseline files are preserved.
Install `requirements-sam.txt` with `uv pip install --no-deps --python <venv-python> -r requirements-sam.txt`.

```powershell
python scripts/prepare_sam.py
python -m pytest -q
python scripts/run_coco_sam.py --smoke
# Inspect outputs/coco_sam/geometry_debug before continuing.
python scripts/run_coco_sam.py --resume
python scripts/evaluate_coco_sam.py
python scripts/report_coco_sam.py
```

Inference accepts only RGB images and saved attribution maps. Evaluation loads
GT after inference completes. Primary metrics use the same CLIP crop as the
baseline; original-image metrics are separate. All candidates, selection scores,
paired comparisons, subgroup results and 50 example panels are saved under
`outputs/coco_sam/`. See `RESULTS.md` and `examples.html` there.

## Saved-artifact SAM reliability diagnostics

`SAM_DIAGNOSTICS_PLAN.md` fixes the diagnostic analysis; `SAM_DIAGNOSTIC_RESULTS.md`
summarizes findings. These commands reuse existing artifacts without CLIP/SAM inference:

```powershell
python scripts/sam_diagnostic_features.py
python scripts/analyze_sam_diagnostics.py
python scripts/sam_diagnostic_oracles.py
python scripts/report_sam_diagnostics.py
```

Outputs are in `outputs/sam_diagnostics/DIAGNOSTIC_REPORT.md`. Features and GT
labels are separate. Models are diagnostic grouped-CV predictors only; no gate
is implemented. Oracle results use GT and cannot be deployed. Previous COCO
baseline and SAM output hashes are checked unchanged.

## Frozen smart SAM selection and fallback

The subsequent experiment treats those 500 images as development data, fits a
small linear candidate ranker, freezes a density-based fallback, and evaluates
500 new image-disjoint COCO images. See [SMART_SAM_RESULTS.md](SMART_SAM_RESULTS.md)
for development and heldout comparisons, limitations, subgroups and oracles.
Exact commands and environment notes are in
[SMART_SAM_REPRODUCTION.md](SMART_SAM_REPRODUCTION.md). Fitted parameters and
source hashes are under `outputs/smart_sam/frozen/`; the full heldout report and
24 representative panels are under `outputs/smart_sam/heldout/`.
