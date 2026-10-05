"""Run only the requested RefCOCOg CS/GE inference, sequentially on GPU."""
import _bootstrap
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path('outputs/cross_attribution_generalization/refcocog')
    for source in ['CS', 'GE']:
        dest = root / source
        dest.mkdir(parents=True, exist_ok=True)
        for stage in ['--smoke', '--resume']:
            state_path = dest / 'run.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get('status') == 'inference_complete':
                break
            if stage == '--smoke' and state.get('technical_smoke_passed'):
                continue
            print('Starting refcocog', source, stage, flush=True)
            with (dest / ('process_' + stage[2:] + '.log')).open('a') as log:
                subprocess.run(
                    [sys.executable, '-u', 'scripts/run_cross_attribution.py',
                     'refcocog', source, stage],
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )
            print('Finished refcocog', source, stage, flush=True)
    print('RefCOCOg CS/GE inference complete; no GT evaluation run.', flush=True)


if __name__ == '__main__':
    main()
