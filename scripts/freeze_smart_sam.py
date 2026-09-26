"""Seal the development-chosen method before any new cohort is constructed."""
import _bootstrap
import json,time,shutil
from pathlib import Path
import pandas as pd
import yaml
from src.utils import write_json,sha256
from smart_sam_core import FEATURES,RANK_FEATURES,GATE_FEATURES

def main():
    out=Path('outputs/smart_sam');frozen=out/'frozen';dev=out/'development'
    assert not (out/'heldout').exists(),'Freeze must precede heldout selection'
    assert not frozen.exists(),'Freeze already exists'
    cv=pd.read_csv(dev/'cv_results.csv');eligible=cv[(cv.selector.isin(['sam_score','smart']))]
    winner=eligible.sort_values('iou',ascending=False).iloc[0]
    assert winner.selector=='smart' and winner.fallback=='density_threshold'
    (frozen/'model_files').mkdir(parents=True)
    models=json.loads((dev/'fitted_models.json').read_text());write_json(frozen/'model_files/models.json',models)
    config={'final_selector':'smart','final_gate':'density_threshold','heldout_seed':2026,'heldout_images':500,
      'targets_per_image':2,'top_k':3,'min_point_distance_crop_pixels':32,'sam_multimask_candidates':3,
      'prompt_template':'a photo of a {class_name}','features':FEATURES,'rank_features':RANK_FEATURES,'gate_features':GATE_FEATURES,
      'density_log_threshold':models['density_thresholds']['smart'],'epsilon':1e-12,'logistic_threshold':.5,
      'ranker':'standardized pairwise logistic C=1 no intercept','gate':'log density threshold from OOF development deciles maximizing IoU',
      'tie_break':'first candidate index','attribution':'unchanged refined full','precision':'fp32',
      'threshold_rule':'unchanged per-image mean','primary_geometry':'unchanged CLIP crop',
      'min_crop_pixels':502,'min_retained_fraction':.25,'bootstrap_seed':2026,'bootstrap_resamples':2000,
      'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(frozen/'config.json',config);(frozen/'frozen_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    sources=[Path('SMART_SAM_PLAN.md'),Path('config_refined.yaml'),
      *[Path('scripts')/p for p in ['smart_sam_core.py','develop_smart_sam.py','run_smart_sam_heldout.py','select_smart_sam_heldout.py','sam_inference_core.py','run_refinement.py','coco_dense_common.py','evaluate_coco_sam.py']],
      *Path('src').glob('*.py'),Path('tests/test_smart_sam.py'),Path('data/checkpoints/ViT-B-16.pt'),Path('data/checkpoints/sam_vit_b_01ec64.pth')]
    import segment_anything,clip
    sources+=list(Path(segment_anything.__file__).parent.rglob('*.py'))+list(Path(clip.__file__).parent.rglob('*.py'))
    # Verify attribution and SAM source implementations against their earlier experiment signatures.
    old=json.loads(Path('outputs/refined_all/run.json').read_text())
    for p,d in old['source_hashes'].items():assert sha256(p)==d,p
    previous=json.loads(Path('outputs/coco_sam/run.json').read_text())
    for p,d in previous['signature']['source_hashes'].items():assert sha256(p)==d,p
    document=f'''# Frozen smart SAM method

Frozen UTC: {config['frozen_utc']}. No heldout cohort exists at freeze time.

Selected method: unchanged refined CLIP attribution -> three positive points -> unchanged SAM ViT-B three candidates -> nine-feature linear pairwise logistic ranker -> density fallback. Use selected SAM iff log((density_inside+1e-12)/(density_outside+1e-12)) >= {config['density_log_threshold']:.17g}; otherwise retain the exact original CLIP binary mask. Argmax ties use first candidate. Candidate features are measured on original-resolution masks, attribution zero outside the visible crop. Primary scoring projects masks into the original CLIP crop.

Why chosen: five-fold image-grouped development mIoU: SAM score 0.400482; smart selector 0.441898; smart+density fallback 0.446508; smart+logistic fallback 0.438331. The simple threshold gives the highest observed out-of-fold mIoU. Logistic classification accuracy did not translate into a better segmentation decision, and its results remain disclosed. No search beyond the predeclared models and nine density deciles plus always/never. Development is exploratory, not final evidence.

Ranker is fitted to all development candidate pairs after validation. Fallback threshold is calibrated using out-of-fold smart-selected masks, never in-sample fitted selector outcomes. Inner four-fold cross-fitting trains gates within each outer development fold. Final comparison logistic gates use all outer out-of-fold choices. Candidate rows and both prompts always remain grouped by image. Final model files contain feature order, means, scales, coefficients, intercepts and thresholds in human-readable JSON.

All 18 extracted feature definitions and nine ranker / 14 logistic-gate features are in frozen_config.yaml and hashed scripts/smart_sam_core.py. Pair-level context cancels in an additive within-pair ranker, so it is reserved for fallback. Raw density, ratio and duplicate point counts remain diagnostic columns rather than redundant learned predictors. No GT-derived or category-ID features are used.

Heldout seed 2026; 500 fresh COCO val2017 images excluding all 500 development IDs; same minimum 502 crop pixels and 25% retained area, two eligible targets each. Freeze precedes cohort construction. All inference outputs saved before any GT evaluation. Frozen smart+density is the sole final proposed method; learned logistic controls remain predeclared comparisons. No method changes based on heldout results.

Checkpoints: CLIP ViT-B/16 SHA256 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f; SAM ViT-B SHA256 ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912. No CLIP/SAM weights are trained. Full source, checkpoint, model and config hashes: integrity.json. Protocol: ../../../SMART_SAM_PLAN.md.
'''
    (frozen/'FROZEN_METHOD.md').write_text(document,encoding='utf-8')
    sources+=list(frozen.rglob('*'));sources+=[dev/'cv_results.csv',dev/'oof_predictions.csv',dev/'fold_audit.json']
    hashes={str(p):sha256(p) for p in sources if p.is_file()};write_json(frozen/'integrity.json',{'frozen_utc':config['frozen_utc'],'hashes':hashes})
    (out/'FROZEN_METHOD.md').write_text(document,encoding='utf-8')
    shutil.copyfile(frozen/'frozen_config.yaml',out/'frozen_config.yaml');shutil.copyfile(frozen/'model_files/models.json',out/'model.json')
    shutil.copyfile(dev/'ablation_results.csv',out/'development_results.csv')
    print(f"Frozen smart selector + log-density threshold {config['density_log_threshold']:.8f}; {len(hashes)} protected sources/artifacts",flush=True)

if __name__=='__main__':main()
