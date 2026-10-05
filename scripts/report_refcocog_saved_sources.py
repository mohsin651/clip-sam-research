"""Requested RefCOCOg tables from saved evaluations; no inference or fitting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import _bootstrap
import json
import numpy as np
import pandas as pd
from report_cross_attribution import read_cell, candidate_rows, paired, SOURCE_NAMES
from report_refcoco_frozen import ImageBootstrap
from pathlib import Path

out = Path('outputs/cross_attribution_generalization')
summary, comparisons, candidates = [], [], []
for source in ['CDA', 'CS', 'GE']:
    f, c = read_cell('refcocog', source)
    assert f is not None
    f = f[f.frame == 'original'].copy()
    assert len(f) == 9602 * 4
    assert f.image_id.nunique() == 2600
    if source != 'CDA':
        audit = json.loads((out/'refcocog'/source/'evaluation_audit.json').read_text())
        assert audit['expressions'] == 9602 and audit['all_predictions_verified_before_gt']
    for cohort, cf in [('standard', f), ('strict_nonoverlap', f[f.strict_nonoverlap])]:
        assert not cf.empty
        boot = ImageBootstrap(cf.image_id)
        key = {'source': source, 'cohort': cohort}
        comparisons.extend(paired(cf, boot, key))
        for method, g in cf.groupby('method', sort=False):
            values = np.column_stack([g.iou, g.correct_instance, (g.delta_iou < -1e-8).astype(float)])
            ci = boot.intervals(g, values)
            row = {**key, 'method': method, 'expressions': len(g), 'images': g.image_id.nunique()}
            for j, metric in enumerate(['iou', 'correct_instance', 'harm']):
                row.update({metric: float(values[:, j].mean()), metric+'_ci_low': float(ci[0, j]), metric+'_ci_high': float(ci[1, j])})
            summary.append(row)
        if source != 'CDA':
            diag = candidate_rows(cf, c[(c.frame == 'original') & c.sent_id.isin(cf.sent_id)])
            metrics = ['naive_accuracy', 'smart_accuracy', 'candidate_oracle', 'naive_regret', 'smart_regret']
            ci = boot.intervals(diag, diag[metrics])
            row = dict(key)
            for j, metric in enumerate(metrics):
                row.update({metric: float(diag[metric].mean()), metric+'_ci_low': float(ci[0, j]), metric+'_ci_high': float(ci[1, j])})
            candidates.append(row)
result = {'summary': summary, 'paired': comparisons, 'candidates': candidates,
          'bootstrap': {'resamples': 2000, 'seed': 2026, 'unit': 'whole image', 'weighting': 'expression macro'}}
target = out/'refcocog_saved_source_statistics.json'
assert not target.exists(), 'Do not overwrite completed statistics'
target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
for name, rows in [('summary', summary), ('paired', comparisons), ('candidates', candidates)]:
    pd.DataFrame(rows).to_csv(out/('refcocog_saved_source_'+name+'.csv'), index=False)
print(json.dumps(result))
