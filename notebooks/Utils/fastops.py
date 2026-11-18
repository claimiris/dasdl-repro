"""
dasdl-repro: vectorised drop-in equivalents of three original routines.

They return the same numbers as the originals (checked by tests/test_fastops.py) but avoid
Python double loops / list building, which is what makes the 800 x 14,999 SAFOD records feasible
in Colab memory and time.  The originals are untouched and still used for the FORGE runs when
`fast=False` is requested.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def patch_fast(A, l1, l2, o1, o2, dtype=None):
    """Same output as Utils.Utils.patch (same zero padding, same patch order, row-major flattening)."""
    A = np.asarray(A)
    if dtype is not None:
        A = A.astype(dtype, copy=False)
    n1, n2 = A.shape
    p1 = (o1 - np.mod(n1 - l1, o1)) % o1
    p2 = (o2 - np.mod(n2 - l2, o2)) % o2
    if p1 or p2:
        A = np.pad(A, ((0, p1), (0, p2)))  # zeros at the end, as in the original
    V = sliding_window_view(A, (l1, l2))[::o1, ::o2]          # (n_i1, n_i2, l1, l2), loop order i1 then i2
    return V.reshape(-1, l1 * l2)                              # copies -> contiguous (N, l1*l2)


def mf_fast(D, nfw=7, ifb=1, axis=2):
    """Same output as Utils.mf.mf (median filter with mirrored (ifb=1) or zero padding)."""
    if np.mod(nfw, 2) == 0:
        nfw = nfw + 1
    D = np.asarray(D, dtype=float)
    if axis == 2:
        D = D.T
    h = (nfw - 1) // 2
    if ifb == 1:
        Dp = np.concatenate((np.flipud(D[0:h, :]), D, np.flipud(D[D.shape[0] - h:, :])), axis=0)
    else:
        Dp = np.concatenate((np.zeros((h, D.shape[1])), D, np.zeros((h, D.shape[1]))), axis=0)
    out = np.empty_like(D)
    step = 2000  # chunk over columns to bound memory
    for j in range(0, D.shape[1], step):
        W = sliding_window_view(Dp[:, j:j + step], nfw, axis=0)  # (n1, ncol, nfw)
        out[:, j:j + step] = np.median(W, axis=-1)
    return out.T if axis == 2 else out


def cwt_scale(x, scale, wavelet='mexh'):
    """Single scale of py_cwt2d.cwt_2d (each scale is computed independently there, so this is
    identical to cwt_2d(x, scales, wavelet)[0][:, :, i] for scales[i] == scale)."""
    from py_cwt2d import cwt_2d
    c, _ = cwt_2d(x, [scale], wavelet)
    return c[:, :, 0]
