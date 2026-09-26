import numpy as np
from sklearn.metrics import average_precision_score

METRICS = ("pg", "epg", "pacc", "ap", "iou")


def localization_metrics(raw, gt, valid=None, threshold=0.5):
    raw, gt = np.asarray(raw, dtype=np.float64), np.asarray(gt, dtype=bool)
    valid = np.ones_like(gt) if valid is None else np.asarray(valid, dtype=bool)
    if raw.shape != gt.shape or valid.shape != gt.shape or not valid.any():
        raise ValueError("Mismatched masks or no valid pixels")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Attribution must be finite and nonnegative")
    scores, labels = raw[valid], gt[valid]
    spread = scores.max() - scores.min()
    norm = (scores - scores.min()) / spread if spread > 0 else np.zeros_like(scores)
    pred = norm >= threshold
    union = (pred | labels).sum()
    energy = scores.sum()
    # No evidence => PG miss, not a lucky top-left hit.
    return {"pg": float(labels[scores.argmax()]) if energy > 0 and spread > 0 else 0.,
            "epg": float(scores[labels].sum() / energy) if energy > 0 else 0.,
            "pacc": float((pred == labels).mean()),
            "ap": float(average_precision_score(labels, scores)) if labels.any() else 0.,
            "iou": float((pred & labels).sum() / union) if union else 1.}


def estimate(values, seed=2026, repeats=2000):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Cannot summarize missing/nonfinite values")
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(repeats)])
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x)>1 else 0.,
            "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))}
