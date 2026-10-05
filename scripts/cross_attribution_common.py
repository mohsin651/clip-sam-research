"""Shared paths and freeze verification; no annotation reads."""
import _bootstrap
import json
from pathlib import Path
from src.utils import sha256
from run_smart_sam_heldout import verify_freeze
from run_clip_surgery_external import file_hashes
import clip_surgery_external_core as cs_core
import grad_eclip_external_core as ge_core

OUT=Path('outputs/cross_attribution_generalization')
DATASETS={'refcoco':(1500,10752),'refcocoplus':(1500,10615),'refcocog':(2600,9602)}
SOURCES=['CS','GE']
def prior(dataset): return Path('outputs')/(dataset+'_frozen')
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def signature():
    for core in [cs_core,ge_core]:
        old=load(core.OUT/'signature.json')
        assert core.source_hashes()==old['official_hashes']
        assert file_hashes(old['adapter_hashes'])==old['adapter_hashes']
    files=[Path('CROSS_ATTRIBUTION_PLAN.md'),Path('scripts/cross_attribution_common.py'),Path('scripts/run_cross_attribution.py')]
    for dataset in DATASETS:
        files += [prior(dataset)/f for f in ['inference_manifest.json','image_manifest.json']]
    return {'files':file_hashes(files),'casr_freeze':verify_freeze(),
      'official_signatures':{s:sha256(c.OUT/'signature.json') for s,c in [('CS',cs_core),('GE',ge_core)]},
      'checkpoints':file_hashes([Path('data/checkpoints')/f for f in ['ViT-B-16.pt','sam_vit_b_01ec64.pth']])}
