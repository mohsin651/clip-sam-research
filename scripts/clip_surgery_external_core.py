"""Thin adapter around unchanged official CS code and unchanged CASR features."""
import _bootstrap
import importlib.util,sys,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
from src.utils import sha256
REPO=Path('data/CLIP-Surgery-source').resolve()
OUT=Path('outputs/external_baselines/clip_surgery')
HELD=Path('outputs/smart_sam/heldout')
COMMIT='d4696d47f49cfe70f49140afe5eb94f94c5f59bc'

def official():
    if 'official_cs' not in sys.modules:
        spec=importlib.util.spec_from_file_location('official_cs',REPO/'clip/__init__.py',submodule_search_locations=[str(REPO/'clip')])
        module=importlib.util.module_from_spec(spec);sys.modules['official_cs']=module;spec.loader.exec_module(module)
    return sys.modules['official_cs']

def source_hashes():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip()==COMMIT
    return {str(p.relative_to(REPO)):sha256(p) for p in REPO.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}

def preprocessing():
    return Compose([Resize((512,512),interpolation=InterpolationMode.BICUBIC),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])

def crop_map(similarity,geometry):
    # Official bilinear interpolation in full-frame resized coordinates, then exact crop.
    m=official().get_similarity_map(similarity[None,:,None],(geometry.rh,geometry.rw))[0,:,:,0]
    return m[geometry.top:geometry.top+224,geometry.left:geometry.left+224].cpu().numpy().copy()

def text_features(model,manifest,device='cuda'):
    names=sorted({t['category'] for r in manifest for t in r['targets']})
    cs=official()
    with torch.no_grad():
        features=cs.encode_text_with_prompt_ensemble(model,names,device)
        redundant=cs.encode_text_with_prompt_ensemble(model,[''],device)
    return dict(zip(names,features)),redundant
