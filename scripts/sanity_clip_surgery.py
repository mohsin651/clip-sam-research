"""Run the unmodified official demo; only IO paths and headless display are redirected."""
import _bootstrap
import os,sys,runpy,time,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.utils import sha256,write_json,environment
ROOT=Path.cwd();REPO=ROOT/'data/CLIP-Surgery-source';OUT=ROOT/'outputs/external_baselines/clip_surgery/sanity';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'sanity.json').exists(),'Do not overwrite official demo sanity run'
sys.path.insert(0,str(REPO))
import clip
assert Path(clip.__file__).resolve().parent==REPO/'clip'
source={str(p.relative_to(REPO)):sha256(p) for p in REPO.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
start=time.perf_counter();original_load=clip.load
# Keep every official model name and setting, only redirect existing weights.
def load(*args,**kwargs):
    kwargs['download_root']=str(ROOT/'data/checkpoints');return original_load(*args,**kwargs)
clip.load=load
from segment_anything import sam_model_registry
original_h=sam_model_registry['vit_h']
def build_h(*args,**kwargs):
    assert Path(kwargs['checkpoint']).name=='sam_vit_h_4b8939.pth'
    kwargs['checkpoint']=str(ROOT/'data/checkpoints/sam_vit_h_4b8939.pth');return original_h(*args,**kwargs)
sam_model_registry['vit_h']=build_h
stats=[];original_map=clip.get_similarity_map

def checked_map(*args,**kwargs):
    m=original_map(*args,**kwargs);assert torch.isfinite(m).all()
    flat=m.reshape(m.shape[0],-1,m.shape[-1]);spans=flat.max(1).values-flat.min(1).values;assert (spans>0).all()
    stats.append({'shape':list(m.shape),'minimum_span':float(spans.min()),'finite':True});return m
clip.get_similarity_map=checked_map
figures=[]
def show(*args,**kwargs):
    p=OUT/f'official_demo_{len(figures):02d}.png';plt.savefig(p,bbox_inches='tight');plt.close();figures.append(str(p.relative_to(OUT)))
plt.show=show
os.chdir(REPO)
try: result=runpy.run_path(str(REPO/'demo.py'),run_name='__main__')
finally:os.chdir(ROOT)
assert len(figures)>0 and len(stats)>0
for p,d in source.items():assert sha256(REPO/p)==d
write_json(OUT/'sanity.json',{'official_demo_completed':True,'official_source_unchanged':True,'headless_figures':figures,'map_checks':stats,'seconds':time.perf_counter()-start,'environment':environment(),'cv2_version':__import__('cv2').__version__,'numpy_version':np.__version__,'sam_h_sha256':sha256(ROOT/'data/checkpoints/sam_vit_h_4b8939.pth'),'io_only_redirects':['CLIP checkpoint directory','SAM H checkpoint path','matplotlib show to PNG']})
print('Official demo completed',len(figures),'figures',flush=True)
