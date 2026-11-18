"""
The proposed workflow exactly as in notebooks/DASDL_Main.ipynb (upstream), wrapped as a function so it can
be applied to every dataset in the paper.

    BP   = bandpass(dn, dt, flo, fhi, 6, 6, 0, 0)
    BPMF = mf(BP, 5, 1, 1)
    CWT  = |cwt_2d(BPMF, arange(1, 10.5, 0.5), 'mexh')[..., -1]|     (finest = 19th scale)
    outB = Train(CWT, BP, epochs=100, batch=128, 48, 48, 8, 8, D1=1024)
    d2   = outB - fkdip(outB, 0.02)
"""
import time
import numpy as np

from Utils.bp import bandpass
from Utils.mf import mf
from Utils.fk import fkdip
from Utils.Utils import patch
from Utils.fastops import patch_fast, mf_fast, cwt_scale

# Upstream notebook settings (FORGE). The paper states the same patch size / network for every example.
UPSTREAM = dict(w1=48, w2=48, s1z=8, s2z=8, D1=1024, epochs=100, batch_size=128)
SCALES = np.arange(1, 10.5, 0.5)   # 19 scales; the 19th (10.0) is the guide


def preprocess(dn, dt, flo, fhi, fast=True):
    BP = bandpass(dn, dt, flo, fhi, 6, 6, 0, 0)
    BPMF = (mf_fast if fast else mf)(BP, 5, 1, 1)
    if fast:
        CWTSCALE = np.abs(cwt_scale(BPMF, SCALES[-1], 'mexh'))
    else:
        import py_cwt2d
        coeffs_nsq, _ = py_cwt2d.cwt_2d(BPMF, SCALES, 'mexh')
        CWTSCALE = np.abs(coeffs_nsq[:, :, -1])
    return BP, BPMF, CWTSCALE


def run_proposed(dn, dt, flo, fhi, tag, fast=True, large=False, train_ratio=1.0, seed=0,
                 model_dir='./models', verbose=2, **overrides):
    """Returns dict with BP, BPMF, CWT, dl (network output before dip filter), den (final), timings, extras."""
    from Utils.DLModel import Train
    p = dict(UPSTREAM); p.update(overrides)
    t0 = time.time()
    BP, BPMF, CWT = preprocess(dn, dt, flo, fhi, fast=fast)
    t_pre = time.time() - t0
    kw = dict(train_ratio=train_ratio, seed=seed, model_dir=model_dir, verbose=verbose, return_extras=True)
    if large:   # SAFOD: identical patches, built without Python lists and stored as float32
        kw.update(patch_fn=lambda A, l1, l2, o1, o2: patch_fast(A, l1, l2, o1, o2, dtype=np.float32), dtype=np.float32,
                  batch_feed=True)
    elif fast:
        kw.update(patch_fn=patch_fast)
    t1 = time.time()
    outB, extras = Train(CWT, BP, p['epochs'], p['batch_size'], p['w1'], p['w2'], p['s1z'], p['s2z'], p['D1'], tag, **kw)
    t_dl = time.time() - t1
    d2 = outB - fkdip(outB, 0.02)
    extras.update(t_preprocess=t_pre, t_train_predict=t_dl, t_total=time.time() - t0)
    return dict(BP=BP, BPMF=BPMF, CWT=CWT, dl=outB, den=d2, extras=extras)
