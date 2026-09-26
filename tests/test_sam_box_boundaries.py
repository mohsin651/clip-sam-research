import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sam_inference_core import Geometry

def test_original_edge_roundoff_is_clipped():
    g=Geometry.create(640,478)
    box=g.box_to_original([30,155,224,224])
    assert box[3]==478 and box[2]<=640
