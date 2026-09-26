"""Write a concise study report from saved measurements only."""
import _bootstrap
import json
from pathlib import Path
import pandas as pd


def main():
    root=Path('outputs/experiment_500')
    meta=json.loads((root/'run.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    frame=pd.read_csv(root/'per_image_results.csv')
    metrics=['pg','epg','pacc','ap','iou']
    lines=['# Small reproduction study', '',
           f"Completed {meta['subset_size']} fixed ImageNet-S919 validation images, all three variants, seed 42.",
           f"Device: {meta['gpu']}. Checkpoint: {meta['checkpoint']}. Attention mode: {meta['config']['attention_mode']}.",
           '', '## Measured results', '',
           '| Variant | PG | EPG | PAcc | AP | mIoU |', '|---|---:|---:|---:|---:|---:|']
    for variant in ['baseline','gcls','full']:
        values=summary['variants'][variant]
        lines.append('| '+variant+' | '+' | '.join(f"{values[m]['mean']:.4f}" for m in metrics)+' |')
    lines+=['','## Paired uncertainty','','All intervals are image-level percentile bootstrap 95% intervals (2,000 resamples, seed 2026).',
            '', '| Comparison | Metric | Mean difference | 95% interval |', '|---|---|---:|---|']
    for comparison, values in summary['paired_differences'].items():
        for metric,s in values.items():
            lines.append(f"| {comparison} | {metric} | {s['mean']:+.5f} | [{s['ci_low']:+.5f}, {s['ci_high']:+.5f}] |")
    first=frame.drop_duplicates('image_id')
    delta=summary['paired_differences']['full_minus_baseline']
    positive=[m for m in metrics if delta[m]['mean']>0]
    negative=[m for m in metrics if delta[m]['mean']<0]
    lines+=['','## Interpretation','',
        'This study tests the explicitly documented full-channel interpretation. It does not establish exact reproduction of the paper.',
        'Baseline and Grad-Only are theoretically proportional in this single-head graph; their scale-invariant metrics cannot reproduce the distinct Table 5 rows. This mathematical limitation is unit-tested.',
        'Their measured PG differs on one of 500 images: FP32 interpolation moves an almost-tied edge maximum across the GT boundary. The max-normalized maps differ by only 4.2e-7 on that sample (ILSVRC2012_val_00014552). This is not evidence of a channel-weighting improvement.',
        'Full spatial weighting improves the measured mean of: '+(', '.join(positive) or 'none')+'.',
        'Full spatial weighting lowers the measured mean of: '+(', '.join(negative) or 'none')+'.',
        'Use the paired intervals above to assess uncertainty. No parameters were adjusted to match published values.',
        'The printed Eq. (11) collapses space to a scalar; our patchwise interpretation and CLS inclusion are documented in REPRODUCTION_NOTES.md. Author clarification is required to settle these discrepancies.',
        '', '## Validation and limitations','',
        f"Zero maps: {int(frame.zero_map.sum())}; constant maps: {int(frame.constant_map.sum())}; failed rows: {int(frame.error.notna().sum())}.",
        f"Absent original target masks: {int((~first.target_present_original).sum())}; absent target after crop: {int((~first.target_present_crop).sum())}. These samples were retained.",
        f"Total experiment wall time: {meta['total_runtime_seconds']:.1f} seconds; average shared attribution time: {frame.runtime_ms.mean():.1f} ms/image.",
        'The prior 20-image smoke run checked all variants and three correct-versus-different-prompt map pairs.',
        'Mirror masks were checked pixel-for-pixel against the official archive; all original image classes were checked against the official source mapping. Original image bytes are obtained from an unofficial mirror.',
        'Evaluation uses the CLIP crop, target-class binary masks, void exclusion and a configurable 0.5 min-max threshold. These protocol choices are not fully specified by the paper.',
        '', 'See outputs/experiment_500/summary.json for per-variant SD and confidence intervals, and run.json for complete provenance.',
        'Launch the saved-results dashboard with `python scripts/run_dashboard.py`.']
    Path('RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__': main()
