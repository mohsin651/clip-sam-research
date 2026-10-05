"""Evaluate the four completed cells explicitly requested by the user.

Reuse the existing evaluator verbatim except its six-cell scheduling barrier.
No inference, prediction edits, or metric/protocol changes.
"""
import _bootstrap
from pathlib import Path
import sys

path = Path(__file__).with_name('evaluate_cross_attribution.py')
source = path.read_text(encoding='utf-8')
old = "for d in DATASETS:\n        for s in SOURCES:"
assert source.count(old) == 1
source = source.replace(old, "for d in ['refcoco', 'refcocoplus']:\n        for s in SOURCES:")
source = source.replace("'all_six_cells_terminal_before_gt':True", "'all_six_cells_terminal_before_gt':False,'requested_four_cells_complete_before_gt':True")
namespace = {'__name__': 'evaluation_only', '__file__': str(path)}
exec(compile(source, str(path), 'exec'), namespace)
assert sys.argv[1] in ['refcoco', 'refcocoplus']
assert sys.argv[2] in ['CS', 'GE']
namespace['main'](sys.argv[1], sys.argv[2])
