"""Fetch the public mirror, verify official IDs, extract selected or all records."""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import _bootstrap
import argparse
import io
import json
from pathlib import Path
import requests
import pyarrow.parquet as pq
from PIL import Image
from src.dataset import ImageNetSegmentation
from src.utils import sha256, write_json


def download_file(url, path):
    partial = path.with_suffix(".partial")
    failures = 0
    while failures < 8:
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            end = offset + 8 * 1024 * 1024 - 1
            with requests.get(url, headers={"Range": f"bytes={offset}-{end}"},
                              stream=True, timeout=(20, 20)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError("Dataset CDN must support bounded byte ranges")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned incorrect resume offset")
                total = int(content_range.rsplit("/", 1)[1])
                with partial.open("ab") as f:
                    for chunk in response.iter_content(65536):
                        f.write(chunk)
            if partial.stat().st_size == total:
                partial.replace(path)
                return
            if partial.stat().st_size > total:
                raise ValueError("Downloaded more bytes than declared")
            if offset // (64*1024*1024) != partial.stat().st_size // (64*1024*1024):
                print(f"  {partial.stat().st_size / 1024**2:.0f}/{total / 1024**2:.0f} MiB", flush=True)
        except requests.RequestException as exc:
            failures += 1
            print(f"Download retry {failures}/8: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"Unable to download {url}; partial data retained")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", default="500")
    args = p.parse_args()
    root = Path("data/ImageNetS919")
    dataset = ImageNetSegmentation(root)
    out = Path("outputs") / ("experiment_" + args.num_samples)
    selected = set(dataset.subset(args.num_samples, out / "subset_ids.txt"))
    repo = "braceletboy/imagenet-s"
    print("Resolving public mirror revision", flush=True)
    revision = requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=30).json()["sha"]
    expected = {f["path"]: f.get("lfs", {}).get("oid") for f in
                requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true", timeout=30).json()}
    manifest = {"source": repo, "revision": revision, "unofficial_mirror": True,
                "official_source": "https://github.com/LUSSeg/ImageNet-S", "files": [],
                "extracted": [], "verification": "official IDs and dimensions; see verification below"}
    seen = set()
    for index in range(4):
        name = f"data/validation-{index:05d}-of-00004.parquet"
        print(f"Downloading/checking {name}", flush=True)
        path = Path("data/downloads/mirror") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Main CDN route is more reliable here; verify bytes against the pinned revision's LFS hash.
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}?download=true"
            download_file(url, path)
        digest = sha256(path)
        if digest != expected[name]:
            raise ValueError(f"Shard checksum differs from pinned revision: {name}")
        manifest["files"].append({"path": name, "sha256": digest})
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                image_id = Path(row["filename"]).stem
                if image_id not in dataset.records or image_id in seen:
                    raise ValueError(f"Unexpected/duplicate mirror ID: {image_id}")
                seen.add(image_id)
                if image_id not in selected:
                    continue
                a, b = dataset.records[image_id]
                for key, rel, folder in [("image", a, "validation"), ("mask", b, "validation-segmentation")]:
                    payload = row[key]["bytes"]
                    if payload is None:
                        raise ValueError("Mirror has external instead of embedded image")
                    dest = root / folder / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(payload)
                sample = dataset[image_id]
                manifest["extracted"].append({"image_id": image_id, "mirror_imagenet_label": row["label"],
                    "image_sha256": sha256(root / "validation" / a),
                    "mask_sha256": sha256(root / "validation-segmentation" / b)})
        print(f"Shard {index + 1}/4 checked; extracted {len(manifest['extracted'])}", flush=True)
    if seen != set(dataset.records) or len(manifest["extracted"]) != len(selected):
        raise ValueError("Mirror does not match complete official validation ID set")
    manifest["verified_official_id_count"] = len(seen)
    manifest["verification"] = "All 12419 IDs match official validation split; selected image/mask sizes and target class validated. Image byte identity to original ImageNet is not independently verified."
    write_json(root / "provenance.json", manifest)
    print(f"Ready: {len(selected)} samples in {root}")


if __name__ == "__main__":
    main()
