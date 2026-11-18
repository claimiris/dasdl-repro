"""Check that the vectorised helpers reproduce the original routines exactly (run from notebooks/: python -m tests.test_fastops)."""
import hashlib, os, sys
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.Utils import patch, patch_inv
from Utils.mf import mf
from Utils.fastops import patch_fast, mf_fast, cwt_scale
import py_cwt2d
from repro import data as D


def main():
    dn = np.array(np.transpose(h5py.File('../data/eq-68.mat', 'r').get('d1')))     # exactly as in DASDL_Main.ipynb
    # 1) the downloaded FORGE EQ68 equals the file shipped with the upstream repo
    d_dl = D.load_forge(68, root='/tmp/dasdl_test_data')
    assert d_dl.shape == dn.shape and np.array_equal(d_dl, dn), 'downloaded eq-68 differs from upstream data/eq-68.mat'
    print('eq-68 download identical to upstream file: OK')

    # 2) patching (both padding branches) and inverse
    for shape in [(2000, 960), (205, 173)]:
        A = np.random.default_rng(0).standard_normal(shape)
        P0, P1 = patch(A, 48, 48, 8, 8), patch_fast(A, 48, 48, 8, 8)
        assert P0.shape == P1.shape and np.array_equal(P0, P1), shape
        assert np.allclose(patch_inv(P1.T, *shape, 48, 48, 8, 8), A)
    print('patch_fast == patch: OK')

    # 3) median filter (on a crop, the original is slow)
    sub = dn[:300, :200]
    for args in [(5, 1, 1), (5, 1, 2), (4, 0, 1)]:
        assert np.array_equal(mf(sub, *args), mf_fast(sub, *args)), args
    print('mf_fast == mf: OK')

    # 4) single CWT scale
    ss = np.arange(1, 10.5, 0.5)
    full, _ = py_cwt2d.cwt_2d(sub, ss, 'mexh')
    assert np.allclose(full[:, :, -1], cwt_scale(sub, ss[-1]), rtol=0, atol=0)
    print('cwt_scale == cwt_2d[..., -1]: OK')


if __name__ == '__main__':
    main()
