import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from .cda_clip import explain
from .dataset import ImageNetSegmentation
from .metrics import METRICS, localization_metrics, estimate
from .model import load_clip_model, CLIP_COMMIT
from .utils import environment, diagnostics, write_json, sha256
from .visualization import resized_map, save_artifacts, save_prompt_comparison, normalize


def summarize(frame, cfg):
    stats, paired = {}, {}
    for variant, group in frame.groupby("variant"):
        stats[variant] = {m: estimate(group[m], cfg["bootstrap_seed"], cfg["bootstrap_replicates"]) for m in METRICS}
    if "full" in stats:
        for other in ("baseline", "gcls"):
            if other not in stats:
                continue
            paired["full_minus_" + other] = {}
            for m in METRICS:
                pivot = frame.pivot(index="image_id", columns="variant", values=m)
                if pivot[["full", other]].isna().any().any():
                    raise ValueError("Paired comparison missing samples")
                paired["full_minus_" + other][m] = estimate(pivot["full"] - pivot[other], cfg["bootstrap_seed"], cfg["bootstrap_replicates"])
    return {"variants": stats, "paired_differences": paired}


def run(smoke=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", default="20" if smoke else "500")
    parser.add_argument("--variant", choices=["all", "baseline", "gcls", "full"], default="all")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output")
    parser.add_argument("--attention-mode", choices=["full", "native"])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.attention_mode:
        cfg["attention_mode"] = args.attention_mode
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    out = Path(args.output or ("outputs/smoke_test" if smoke else f"outputs/experiment_{args.num_samples}"))
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p): sha256(p) for p in sorted(Path("src").glob("*.py"))}
    signature = {"config": cfg, "source_hashes": source_hashes, "clip_commit": CLIP_COMMIT,
                 "metadata_hashes": {str(p): sha256(p) for p in sorted(Path("data/metadata").glob("*")) if p.is_file()}}
    if not smoke:
        gate = Path("outputs/smoke_test/smoke_pass.json")
        if not gate.exists():
            raise RuntimeError("Run the 20-image smoke test first")
        passed = json.loads(gate.read_text())
        if passed["signature"] != signature or passed["n"] < 20:
            raise RuntimeError("Smoke test does not match this code/config or has fewer than 20 samples")
        requested = {"baseline", "gcls", "full"} if args.variant == "all" else {args.variant}
        if not requested.issubset(passed["variants"]):
            raise RuntimeError("Smoke test did not cover all requested variants")
    if (out / "per_image_results.csv").exists():
        raise RuntimeError("Output already contains results; choose a new --output directory")
    if smoke:
        (out / "smoke_pass.json").unlink(missing_ok=True)
    ds = ImageNetSegmentation(cfg["dataset_root"])
    ids = ds.subset(args.num_samples, out / "subset_ids.txt", cfg["seed"])
    # Check data before loading checkpoint.
    for image_id in ids:
        ds[image_id]
    model, preprocess = load_clip_model(cfg["device"], cfg["checkpoint_dir"])
    checkpoint_hash = sha256(Path(cfg["checkpoint_dir"]) / "ViT-B-16.pt")
    run_meta = {**environment(), **signature, "dataset": "ImageNet-S919 validation",
                "subset_size": len(ids), "subset_sha256": sha256(out / "subset_ids.txt"),
                "checkpoint": "OpenAI CLIP ViT-B/16", "checkpoint_sha256": checkpoint_hash,
                "variants": args.variant, "status": "running"}
    provenance = Path(cfg["dataset_root"]) / "provenance.json"
    run_meta["dataset_provenance_sha256"] = sha256(provenance) if provenance.exists() else None
    write_json(out / "run.json", run_meta)
    rows, sensitivity = [], []
    variants = ["baseline", "gcls", "full"] if args.variant == "all" else [args.variant]
    start = time.perf_counter()
    try:
        for index, image_id in enumerate(tqdm(ids)):
            sample = ds[image_id]
            image = preprocess(sample["image"]).unsqueeze(0).to(cfg["device"])
            prompt = cfg["prompt_template"].format(class_name=sample["class_name"])
            torch.cuda.synchronize(); tick = time.perf_counter()
            maps, similarity, debug = explain(model, image, prompt, cfg["attention_mode"])
            torch.cuda.synchronize(); runtime = (time.perf_counter()-tick)*1000
            folder = out / "images" / image_id
            folder.mkdir(parents=True, exist_ok=True)
            if index < 3:
                torch.save(debug, folder / "tensors.pt")
                write_json(folder / "diagnostics.json", diagnostics(debug))
            if any(not torch.isfinite(v).all() for v in debug.values()):
                raise RuntimeError(f"Nonfinite intermediate tensor: {image_id}")
            for variant in variants:
                patch = maps[variant]
                zero = bool((patch == 0).all())
                constant = bool(patch.max() == patch.min())
                raw = resized_map(patch)
                scores = localization_metrics(raw, sample["segmentation_mask"], sample["valid_mask"], cfg["threshold"])
                rows.append({"image_id": image_id, "class_id": sample["class_id"], "class_name": sample["class_name"],
                    "segmentation_class_id": sample["segmentation_class_id"],
                    "variant": variant, "clip_similarity": similarity, **scores, "runtime_ms": runtime,
                    "target_present_original": sample["target_present_original"],
                    "target_present_crop": bool(sample["segmentation_mask"].any()),
                    "runtime_scope": "shared_forward_backward_all_variants", "zero_map": zero,
                    "constant_map": constant, "error": "degenerate map" if zero or constant else ""})
                save_artifacts(folder, sample, patch, scores, similarity, variant, cfg["threshold"])
                if zero or constant:
                    raise RuntimeError(f"Degenerate map: {image_id}/{variant}; inspect diagnostics")
            if smoke and index < 3:
                other_name = "fire engine" if sample["synset"] != "n03345487" else "goldfish"
                other_prompt = cfg["prompt_template"].format(class_name=other_name)
                other, other_score, _ = explain(model, image, other_prompt, cfg["attention_mode"])
                delta = float(np.abs(normalize(maps["full"].numpy()) - normalize(other["full"].numpy())).mean())
                sensitivity.append({"image_id": image_id, "correct_prompt": prompt, "different_prompt": other_prompt,
                                    "mean_absolute_normalized_difference": delta, "different_cosine": other_score})
                save_prompt_comparison(folder / "prompt_sensitivity.jpg", sample["view"], maps["full"], other["full"], prompt, other_prompt)
                if delta < 1e-6:
                    raise RuntimeError("Prompt conditioning did not change attribution")
            pd.DataFrame(rows).to_csv(out / "per_image_results.csv", index=False)
        frame = pd.DataFrame(rows)
        summary = summarize(frame, cfg)
        write_json(out / "summary.json", summary)
        pd.DataFrame([{"variant": v, "metric": m, **values} for v, metrics in summary["variants"].items()
                      for m, values in metrics.items()]).to_csv(out / "summary.csv", index=False)
        write_json(out / "prompt_sensitivity.json", sensitivity)
        run_meta.update(status="complete", total_runtime_seconds=time.perf_counter()-start,
                        failed_samples=0, zero_map_count=int(frame.zero_map.sum()))
        if smoke:
            write_json(out / "smoke_pass.json", {"signature": signature, "n": len(ids), "variants": variants, "checkpoint_sha256": checkpoint_hash})
        print(frame.groupby("variant")[list(METRICS)].mean().to_string())
    except Exception as exc:
        pd.DataFrame(rows).to_csv(out / "per_image_results.csv", index=False)
        run_meta.update(status="failed", error=str(exc), total_runtime_seconds=time.perf_counter()-start)
        raise
    finally:
        write_json(out / "run.json", run_meta)
