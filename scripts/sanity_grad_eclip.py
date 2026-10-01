"""Execute official image-demo cells; skip installation/magic and unused open_clip import only."""
import _bootstrap
import json,time,os
import numpy as np
import torch
import clip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from src.utils import write_json,environment,sha256
from grad_eclip_external_core import OUT,REPO,cells,source_hashes,load_official

def main():
    start=time.perf_counter();dest=(OUT/'sanity').resolve();dest.mkdir(parents=True,exist_ok=False)
    before=source_hashes();model,scope=load_official()
    scope.update({'clip':clip,'cv2':cv2,'plt':plt,'Image':Image,'os':os})
    figures=[]
    def save():
        name=f'official_demo_{len(figures):02d}.png';plt.gcf().savefig(dest/name,dpi=130);figures.append(name);plt.close()
    plt.show=save
    cwd=os.getcwd()
    try:
        os.chdir(REPO)
        for index in [5,6,7,10]:
            source=''.join(cells()[index]['source']).replace('%matplotlib inline','')
            exec(compile(source,str(REPO/f'grad_eclip_image.ipynb#cell{index}'),'exec'),scope)
        save()
    finally:os.chdir(cwd)
    maps=[m.detach().float().cpu().numpy() for m in scope['grad_emaps']]
    assert all(np.isfinite(m).all() and np.ptp(m)>0 and (m>=0).all() for m in maps)
    changed=float(np.max(np.abs(maps[0]/maps[0].max()-maps[1]/maps[1].max())))
    assert changed>1e-5;assert source_hashes()==before
    np.savez_compressed(dest/'official_maps.npz',maps=np.stack(maps))
    write_json(dest/'sanity.json',{'official_demo_completed':True,'texts':scope['texts'],'map_shape':list(maps[0].shape),
      'preprocessed_shape':list(scope['img_preprocessed_k'].shape),'gradient_enabled':True,'finite_nonconstant_nonnegative':True,
      'dog_car_max_normalized_difference':changed,'source_unchanged':True,'source_hashes':before,'figures':figures,
      'checkpoint_sha256':sha256('data/checkpoints/ViT-B-16.pt'),'environment':environment(),'numpy_version':np.__version__,
      'adjustments':['Skip notebook package installation and inline plotting magic','Provide imports; omit unused open_clip import',
        'Redirect checkpoint directory and headless plotting only','Execute official definition/demo cell text without algorithm edits'],
      'seconds':time.perf_counter()-start})
    print('Official Grad-ECLIP demo passed, including prompt sensitivity',flush=True)

if __name__=='__main__':main()
