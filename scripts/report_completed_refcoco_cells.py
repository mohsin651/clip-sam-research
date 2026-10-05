"""Requested four-cell report from evaluated saved predictions only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import _bootstrap
import json
from pathlib import Path
import pandas as pd
from report_cross_attribution import read_cell, candidate_rows, paired
from report_refcoco_frozen import ImageBootstrap, table

out = Path('outputs/cross_attribution_generalization')
lines = ['NEW CS/GE RefCOCO and RefCOCO+ evaluation', '',
         'Saved inference was reused. No inference, training, tuning or prediction changes.',
         'Evaluation scope: four completed cells, explicitly requested by the user; RefCOCOg was not started.',
         'The separate wrapper changes only the original six-cell scheduling barrier and its audit label.',
         'Metrics and prediction/source/annotation hash verification reuse the existing evaluator.', '',
         'Saved prediction counts: RefCOCO CS=10,752; GE=10,752. RefCOCO+ CS=10,615; GE=10,615.', '']
all_pairs = []
candidate = []
instance = []
harms = []
verdicts = []
for dataset, label, count in [('refcoco', 'RefCOCO', 10752), ('refcocoplus', 'RefCOCO+', 10615)]:
    comparison = []
    for source in ['CDA', 'CS', 'GE']:
        if source != 'CDA':
            audit = json.loads((out/dataset/source/'evaluation_audit.json').read_text())
            assert audit['expressions'] == count and audit['all_predictions_verified_before_gt']
        f, c = read_cell(dataset, source)
        f = f[f.frame == 'original'].copy()
        assert len(f) == count*4
        methods = {m: f[f.method == m].sort_values('sent_id') for m in ['MAP','NAIVE','SMART','FINAL']}
        comparison.append({'Source':source, **{m+' mIoU':float(g.iou.mean()) for m,g in methods.items()},
                           'Naive harm %':100*float((methods['NAIVE'].delta_iou < -1e-8).mean()),
                           'Final harm %':100*float((methods['FINAL'].delta_iou < -1e-8).mean())})
        if source == 'CDA': continue
        boot = ImageBootstrap(f.image_id)
        pairs = paired(f, boot, {'dataset':dataset,'source':source})
        all_pairs.extend(pairs)
        selected = [r for r in pairs if r['comparison'] != 'NAIVE minus MAP' and r['metric'] == 'iou']
        lines.extend([label+' / '+source+' paired mIoU differences', table(pd.DataFrame(selected)[['comparison','difference','ci_low','ci_high']])])
        smart = next(r for r in selected if r['comparison']=='SMART minus NAIVE')
        verdicts.append({'Experiment':label+' / '+source,'Smart transfer':'YES' if smart['difference']>0 and smart['ci_low']>0 else 'NO'})
        h = {'Experiment':label+' / '+source}
        for m in ['NAIVE','SMART','FINAL']: h[m+' harm %'] = 100*float((methods[m].delta_iou < -1e-8).mean())
        h['Naive to final reduction pp'] = h['NAIVE harm %']-h['FINAL harm %']
        harms.append(h)
        diag = candidate_rows(f, c[c.frame=='original'])
        candidate.append({'Experiment':label+' / '+source, **{m:float(diag[m].mean()) for m in ['naive_accuracy','smart_accuracy','candidate_oracle','naive_regret','smart_regret','no_candidate_at_05']}})
        row = {'Experiment':label+' / '+source}
        for m in ['NAIVE','SMART','FINAL']: row[m+' correct %']=100*float(methods[m].correct_instance.mean())
        for comp in ['SMART minus NAIVE','FINAL minus NAIVE']:
            r=next(r for r in pairs if r['comparison']==comp and r['metric']=='correct_instance')
            row[comp+' pp [95% CI]']=f"{100*r['difference']:+.4f} [{100*r['ci_low']:+.4f}, {100*r['ci_high']:+.4f}]"
        instance.append(row)
    lines.extend([label+' three-source full-image mIoU and harm table', table(pd.DataFrame(comparison))])
lines.extend(['Harm against each source\'s own MAP baseline (%)',table(pd.DataFrame(harms)),
              'Candidate selection (accuracy and no-good rates are fractions; oracle/regret are IoU)',table(pd.DataFrame(candidate)),
              'Correct-instance overlap proxy (%) and paired differences (percentage points)',table(pd.DataFrame(instance)),
              'Smart-selection transfer verdicts',table(pd.DataFrame(verdicts)),
              'Protocol: full-original-image expression-macro referred-instance IoU. 2,000 paired whole-image bootstrap resamples, seed 2026; all expressions from a sampled image stay grouped. No multiplicity adjustment. Harm uses tolerance 1e-8 against each source\'s own MAP. Best-candidate accuracy accepts any tied optimum within 1e-8; no-good means candidate oracle IoU < 0.5.',
              'Correct-instance means target IoU exceeds every other same-category instance IoU; it is an overlap proxy. Oracles use GT after inference and are not deployable.',
              'RefCOCO+ overlaps all 1,500 prior RefCOCO test images; no strict non-overlap cohort exists. These are inspected cohorts. MAP/fallback is crop-limited even though primary scoring uses the full original image.',
              'Artifacts: outputs/cross_attribution_generalization/{refcoco,refcocoplus}/{CS,GE}/per_expression_results.csv, candidate_metrics.csv, candidate_analysis_per_expression.csv, evaluation_audit.json.',
              'Paired statistics for this report: outputs/cross_attribution_generalization/completed_refcoco_paired_comparisons.csv.',
              'CDA comparison rows reuse outputs/refcoco_frozen and outputs/refcocoplus_frozen. No RefCOCOg work is included.'])
pd.DataFrame(all_pairs).to_csv(out/'completed_refcoco_paired_comparisons.csv',index=False)
pd.DataFrame(candidate).to_csv(out/'completed_refcoco_candidate_summary.csv',index=False)
target=Path('REFCOCO_AND_REFCOCOPLUS_SHARE_SUMMARY.txt')
target.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
print(target.resolve())
