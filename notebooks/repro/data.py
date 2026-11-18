"""
Download + load the field data used in the paper.

The paper (Saad, Ravasi & Alkhalifah, 2024) cites Lellouch et al. (2021) for both field datasets but gives
no download link.  The exact extracted windows are published (GPL-3) by Yangkang Chen, a co-author of the
benchmark papers, in two data repositories:

  FORGE  : https://github.com/chenyk1990/dasdenoising-dataonly  (mat_raw/eq-N.mat, 2000 x 960, dt = 0.5 ms;
           mat_bpsomffk/eq-N.mat = BP+SOMF+FK result of Chen et al. 2023a, the paper's "SOMF" benchmark)
  SAFOD  : https://github.com/chenyk1990/dasmrrcoh-dataonly    (raw/<UTC>_mag<M>.mat, 800 x 14,999, dt = 4 ms;
           processed/eqN.mat = BP+MF+MRR result of Chen et al. 2023b, the paper's "MRR" benchmark)

The repository's own data/eq-68.mat is byte-identical to dasdenoising-dataonly/mat_raw/eq-68.mat.
"""
import os
import urllib.request
import numpy as np

FORGE_URL = 'https://raw.githubusercontent.com/chenyk1990/dasdenoising-dataonly/main/{sub}/eq-{n}.mat'
SAFOD_URL = 'https://raw.githubusercontent.com/chenyk1990/dasmrrcoh-dataonly/main/{sub}/{name}'

# SAFOD event numbering follows dasmrrcoh-dataonly/processed/EQNO2RawFileName.txt (the paper's "EQ3"/"EQ18").
SAFOD_EVENTS = {
    3: '2017-06-23T22:03:22.380000Z_mag2.17.mat',
    18: '2017-07-02T03:17:32.300000Z_mag2.86.mat',
}

# FORGE events used in the paper (file names from the paper) and their index in mat_raw/.
# 'verified' = the index<->file mapping is stated in the data repo or the paper's figure label; 'inferred' = see README.
FORGE_EVENTS = {
    'ex1':     dict(eq=68, file='FORGE_78-32_iDASv3-P11_UTC190502024438.sgy', mag=0.065,  status='verified (paper Fig. 3 label "EQ68"; upstream repo uses eq-68)'),
    'ex2':     dict(eq=44, file='FORGE_78-32_iDASv3-P11_UTC190428175708.sgy', mag=-0.087, status='verified (paper Fig. 6 label "EQ44")'),
    'limitA':  dict(eq=3,  file='FORGE_78-32_iDASv3-P11_UTC190423213209.sgy', mag=0.91,   status='verified (dasdenoising-dataonly/test_oneevent.m: "ii=3: ...UTC190423213209.sgy")'),
    'limitB':  dict(eq=45, file='FORGE_78-32_iDASv3-P11_UTC190428180038.sgy', mag=1.07,   status='inferred (no public index; catalog is chronological and this file is 3.5 min after EQ44)'),
    'ablation': dict(eq=68, file='(paper: FORGE_78-32_iDASv3P11_UTC190426080738)', mag=-0.957, status='substituted: no public index for UTC190426080738, block outputs shown for EQ68'),
}


def _fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        print('downloading', url)
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def _loadmat(path, key):
    try:
        import scipy.io as sio
        return np.asarray(sio.loadmat(path)[key])
    except NotImplementedError:          # MATLAB v7.3 (HDF5) file
        import h5py
        with h5py.File(path, 'r') as f:
            return np.array(f[key]).T     # h5py returns the MATLAB array transposed


def load_forge(eq, root='../data', kind='raw'):
    """FORGE record as (time=2000, channel=960). kind='raw' | 'somf' (Chen et al. 2023a BP+SOMF+FK)."""
    sub = {'raw': 'mat_raw', 'somf': 'mat_bpsomffk'}[kind]
    p = _fetch(FORGE_URL.format(sub=sub, n=eq), os.path.join(root, 'forge', sub, f'eq-{eq}.mat'))
    return _loadmat(p, 'd1')              # MATLAB array is (time, channel)


def load_safod(eqno, root='../data', kind='raw'):
    """SAFOD record as (time=14,999, channel=800). kind='raw' | 'mrr' (Chen et al. 2023b BP+MF+MRR)."""
    if kind == 'raw':
        name = SAFOD_EVENTS[eqno]
        local = name.replace(':', '-')        # ':' is not allowed in Windows file names
        p = _fetch(SAFOD_URL.format(sub='raw', name=name), os.path.join(root, 'safod', 'raw', local))
        d = _loadmat(p, 'data')
    else:
        name = f'eq{eqno}.mat'
        p = _fetch(SAFOD_URL.format(sub='processed', name=name), os.path.join(root, 'safod', 'processed', name))
        d = _loadmat(p, 'd_bpmfmrr')
    d = np.asarray(d, dtype=np.float64).T  # stored as (channel=800, time=14,999)
    d[~np.isfinite(d)] = 0.0
    return d
