"""Execute official notebook definitions verbatim; no attribution reimplementation."""
import _bootstrap
import contextlib,io,json,subprocess,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
import clip
from src.utils import sha256

REPO=Path('data/Grad-Eclip-source').resolve()
OUT=Path('outputs/external_baselines/grad_eclip')
HELD=Path('outputs/smart_sam/heldout')
CS=Path('outputs/external_baselines/clip_surgery')
COMMIT='e370e6cb194faf2020f5d1ed268f9d57e91a38e6'

def cells():
    return json.loads((REPO/'grad_eclip_image.ipynb').read_text(encoding='utf-8'))['cells']

def source_hashes():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip()==COMMIT
    names=subprocess.check_output(['git','-C',str(REPO),'ls-files']).decode().splitlines()
    # Ignore the untracked historical Chefer reference; never import it.
    return {name:sha256(REPO/name) for name in names}

def load_official():
    model,preprocess=clip.load('ViT-B/16',device='cuda',download_root=str(Path('data/checkpoints').resolve()))
    scope={'math':math,'torch':torch,'F':F,'T':T,'Compose':Compose,'Resize':Resize,'ToTensor':ToTensor,
        'Normalize':Normalize,'InterpolationMode':InterpolationMode,'np':np,'clipmodel':model,
        'clip_inres':model.visual.input_resolution,'clip_ksize':model.visual.conv1.kernel_size,'preprocess':preprocess}
    for index in [3,4,9]:
        source=''.join(cells()[index]['source'])
        exec(compile(source,str(REPO/f'grad_eclip_image.ipynb#cell{index}'),'exec'),scope)
    return model,scope

def encode(scope,image):
    tensor=scope['imgprocess'](image).cuda().unsqueeze(0)
    with contextlib.redirect_stdout(io.StringIO()):result=scope['clip_encode_dense'](tensor,n=1)
    return result,tensor.shape

def explain(scope,result,text_embedding):
    outputs,last_feat,vs,qs,ks,attns,attn_outputs,map_size=result
    cosine=(F.normalize(outputs[:,0],dim=-1)@text_embedding.T)[0]
    assert cosine.requires_grad and attn_outputs[-1].requires_grad
    return [scope['grad_eclip'](c,qs,ks,vs,attn_outputs,map_size).detach() for c in cosine],cosine.detach()

def crop_map(patch,geometry):
    # Official visualization Resize operator, applied to the raw nonnegative map.
    # Full-frame resize first, then the original evaluation crop. No new normalization.
    full=T.Resize((geometry.rh,geometry.rw))(patch.unsqueeze(0))[0]
    return full[geometry.top:geometry.top+224,geometry.left:geometry.left+224].cpu().float().numpy().copy()
