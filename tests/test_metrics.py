import numpy as np
import pytest
from src.metrics import localization_metrics, estimate


def test_perfect():
    gt=np.array([[1,0],[0,1]])
    assert localization_metrics(gt,gt)==dict(pg=1.,epg=1.,pacc=1.,ap=1.,iou=1.)


def test_opposite_and_zero():
    gt=np.array([[1,0],[0,1]])
    r=localization_metrics(1-gt,gt)
    assert r==dict(pg=0.,epg=0.,pacc=0.,ap=.5,iou=0.)
    r=localization_metrics(np.zeros((2,2)),gt)
    assert r==dict(pg=0.,epg=0.,pacc=.5,ap=.5,iou=0.)


def test_energy_uses_raw_and_ignores_void():
    raw=np.array([[4,2],[1,100.]])
    gt=np.array([[1,0],[1,0]])
    valid=np.array([[1,1],[1,0]],bool)
    r=localization_metrics(raw,gt,valid)
    assert r['epg']==pytest.approx(5/7)
    assert r['pg']==1 and r['iou']==.5 and r['pacc']==pytest.approx(2/3)


def test_bootstrap_reproducible():
    assert estimate([0,1,1], repeats=100)==estimate([0,1,1], repeats=100)
    assert estimate([2,2,2],repeats=100)['ci_low']==2


def test_invalid():
    with pytest.raises(ValueError):
        localization_metrics([[np.nan]],[[1]])


def test_empty_target_is_retained():
    r=localization_metrics([[1.,0.]],[[0,0]])
    assert r==dict(pg=0.,epg=0.,pacc=.5,ap=0.,iou=0.)
