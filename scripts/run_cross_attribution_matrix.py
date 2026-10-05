"""Sequential GPU cells; no GT scoring until all inference processes terminate."""
import _bootstrap
import sys,subprocess,json,time
from pathlib import Path
OUT=Path('outputs/cross_attribution_generalization')
def main():
    for dataset in ['refcoco','refcocoplus','refcocog']:
        for source in ['CS','GE']:
            dest=OUT/dataset/source;dest.mkdir(parents=True,exist_ok=True)
            for stage in ['--smoke','--resume']:
                state=json.loads((dest/'run.json').read_text()) if (dest/'run.json').exists() else {}
                if state.get('status')=='inference_complete':break
                if stage=='--smoke' and state.get('technical_smoke_passed'):continue
                log=dest/('process_'+stage[2:]+'.log')
                print('Starting',dataset,source,stage,flush=True)
                with log.open('a') as stream:
                    rc=subprocess.call([sys.executable,'scripts/run_cross_attribution.py',dataset,source,stage],stdout=stream,stderr=subprocess.STDOUT)
                if rc:print('STOPPED CELL',dataset,source,'see',log,flush=True);break
                print('Finished',dataset,source,stage,flush=True)
    for dataset in ['refcoco','refcocoplus','refcocog']:
        for source in ['CS','GE']:
            dest=OUT/dataset/source;run=json.loads((dest/'run.json').read_text())
            if run['status']!='inference_complete' or (dest/'evaluation_audit.json').exists():continue
            print('Evaluating',dataset,source,flush=True)
            with (dest/'process_evaluation.log').open('a') as stream:
                subprocess.run([sys.executable,'scripts/evaluate_cross_attribution.py',dataset,source],stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Matrix processes finished',flush=True)
if __name__=='__main__':main()
