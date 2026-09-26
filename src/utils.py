import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0,
            "git_commit": commit,
            "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}


def diagnostics(tensors):
    out = {}
    for key, value in tensors.items():
        a = value.numpy()
        finite = a[np.isfinite(a)]
        out[key] = {"shape": list(a.shape), "min": float(finite.min()) if finite.size else None,
                    "max": float(finite.max()) if finite.size else None,
                    "mean": float(finite.mean()) if finite.size else None,
                    "std": float(finite.std()) if finite.size else None,
                    "nan_count": int(np.isnan(a).sum()), "inf_count": int(np.isinf(a).sum())}
    return out
