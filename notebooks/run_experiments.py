"""
Reproduce every experiment of Saad, Ravasi & Alkhalifah (2024) with the upstream DLDAS_Denoising-pub code.

Run from the `notebooks/` directory (the upstream code imports `Utils.*` relative to it):

    python run_experiments.py --exp ex1 ex2 ratio limitA limitB ex3 ex4
    python run_experiments.py --exp all

Experiments (paper section / figures):
    ex1     FORGE EQ68  - proposed vs SOMF vs FCDenseNet           Figs. 3-5, Fig. 15 (block outputs),
                                                                   Fig. 17 (weights), Sec. 4.4 (timing)
    ex2     FORGE EQ44  - proposed vs SOMF                          Figs. 6-8
    ex3     SAFOD EQ3   - proposed vs MRR (+ trace 70 scalograms)   Figs. 9-11
    ex4     SAFOD EQ18  - proposed vs MRR (+ trace 120 scalograms)  Figs. 12-14
    ratio   FORGE EQ68  - training ratio 10/30/50/70/90 %           Fig. 16, Sec. 4.4
    limitA  FORGE EQ3   - proposed vs FCDenseNet (leakage)          Fig. 18
    limitB  FORGE EQ45* - proposed vs SOMF (residual vertical noise) Fig. 19   (*index inferred, see README)

Outputs: ../results/<exp>/*.png, ../results/<exp>/metrics.json, arrays in ../outputs/<exp>.npz
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro import data as D
from repro.pipeline import run_proposed
from repro import analysis as A
from repro import benchmarks as B

RES, OUT, MOD = '../results', '../outputs', './models'
DT_FORGE, DT_SAFOD = 0.0005, 1 / 250.0


def _dir(exp):
    p = os.path.join(RES, exp); os.makedirs(p, exist_ok=True); os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
    return p


def _save(exp, metrics, **arrays):
    with open(os.path.join(RES, exp, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    np.savez_compressed(os.path.join(OUT, f'{exp}.npz'), **{k: np.asarray(v, np.float32) for k, v in arrays.items()})
    print(json.dumps(metrics, indent=2, default=float))


def _zoom(den, dt, frac_t=0.3, nch=240):
    """Window around the strongest coherent energy of the denoised section (paper's red rectangles are not specified)."""
    e = A.uniform_filter(den ** 2, size=(51, 51))
    i, j = np.unravel_index(np.argmax(e), e.shape)
    nt, nx = den.shape; ht = int(frac_t * nt / 2); hx = nch // 2
    i0 = max(0, min(i - ht, nt - 2 * ht)); j0 = max(0, min(j - hx, nx - 2 * hx))
    return ((i0 * dt, (i0 + 2 * ht) * dt), (j0, j0 + 2 * hx))


def _hist_summary(h):
    return dict(epochs_run=len(h['loss']), first_loss=h['loss'][0], last_loss=h['loss'][-1], best_loss=min(h['loss']))


# ------------------------------------------------------------------------------------------- FORGE
def forge_proposed(eq, tag, **kw):
    dn = D.load_forge(eq)
    r = run_proposed(dn, DT_FORGE, 1e-3, 250, tag, model_dir=MOD, **kw)
    return dn, r


def exp_ex1():
    p = _dir('ex1')
    dn, r = forge_proposed(68, 'forge68')
    somf = D.load_forge(68, kind='somf')
    fcd, fx = B.fcdensenet(r['BP'], tag='forge68', model_dir=MOD)
    panels = [('SOMF (Chen et al. 2023a)', somf), ('FCDenseNet (re-impl.)', fcd), ('Proposed', r['den'])]
    z = _zoom(r['den'], DT_FORGE)
    A.compare_figure(panels, dn, DT_FORGE, f'{p}/fig3-5_ex1_eq68.png', 'FORGE example 1 (EQ68, UTC190502024438)', zoom=z, fmax=300)
    A.compare_figure(panels, dn, DT_FORGE, f'{p}/fig3_ex1_eq68_upstream_colorscale.png',
                     'FORGE example 1 (EQ68) - upstream SymLog display', sym=True, fk=False)
    # Fig. 15: output of each block (on EQ68, see README)
    blocks = [('Band-pass', r['BP']), ('Band-pass + median filter', r['BPMF']), ('DL output', r['dl']), ('Dip filter output (final)', r['den'])]
    A.compare_figure(blocks, dn, DT_FORGE, f'{p}/fig15_blocks_eq68.png', 'Output of each block (EQ68)', fk=True, fmax=300)
    # Fig. 17: weights
    w = r['extras']['dense_weights']
    A.weights_figure([('BP -> encoder 1', w['bp_to_enc1'], 1), ('scale -> encoder 1', w['cwt_to_enc1'], 1),
                      ('last decoder -> output', w['dec_to_out'], 0)], f'{p}/fig17_weights.png', 'Learned weights (48x48)')
    m = {'event': D.FORGE_EVENTS['ex1'], 'zoom_window': z,
         'proposed': {**A.leakage_metrics(dn, r['den']), **_hist_summary(r['extras']['history']),
                      'n_patches': r['extras']['n_patches'], 'fit_seconds': r['extras']['fit_seconds'],
                      't_total_seconds': r['extras']['t_total']},
         'somf_published': A.leakage_metrics(dn, somf),
         'fcdensenet_reimpl': {**A.leakage_metrics(dn, fcd), **_hist_summary(fx['history']), 'fit_seconds': fx['fit_seconds']},
         'paper_timing_rtx2080ti_s': {'proposed_all_patches': 456, 'fcdensenet': 1300}}
    _save('ex1', m, noisy=dn, proposed=r['den'], dl=r['dl'], bp=r['BP'], somf=somf, fcdensenet=fcd)


def exp_ex2():
    p = _dir('ex2')
    dn, r = forge_proposed(44, 'forge44')
    somf = D.load_forge(44, kind='somf')
    panels = [('SOMF (Chen et al. 2023a)', somf), ('Proposed', r['den'])]
    z = _zoom(r['den'], DT_FORGE)
    A.compare_figure(panels, dn, DT_FORGE, f'{p}/fig6-8_ex2_eq44.png', 'FORGE example 2 (EQ44, UTC190428175708)', zoom=z, fmax=300)
    m = {'event': D.FORGE_EVENTS['ex2'], 'zoom_window': z,
         'proposed': {**A.leakage_metrics(dn, r['den']), **_hist_summary(r['extras']['history']), 'fit_seconds': r['extras']['fit_seconds']},
         'somf_published': A.leakage_metrics(dn, somf)}
    _save('ex2', m, noisy=dn, proposed=r['den'], somf=somf)


def exp_ratio():
    p = _dir('ratio')
    dn = D.load_forge(68)
    items_d, items_n, m = [], [], {'event': D.FORGE_EVENTS['ex1'], 'paper_timing_rtx2080ti_s': {'90%': 425, '50%': 242, '100%': 456}}
    arrays = {'noisy': dn}
    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = run_proposed(dn, DT_FORGE, 1e-3, 250, f'forge68_r{int(ratio*100)}', model_dir=MOD, train_ratio=ratio, seed=0)
        items_d.append((f'{int(ratio*100)}% patches', r['den'])); items_n.append((f'{int(ratio*100)}%: removed noise', dn - r['den']))
        m[f'{int(ratio*100)}%'] = {**A.leakage_metrics(dn, r['den']), **_hist_summary(r['extras']['history']),
                                    'n_train_patches': r['extras']['n_train_patches'], 'fit_seconds': r['extras']['fit_seconds']}
        arrays[f'r{int(ratio*100)}'] = r['den']
    clip = np.percentile(np.abs(dn), 99)
    A.grid_figure(items_d + items_n, DT_FORGE, f'{p}/fig16_training_ratio.png', 'Training ratio (EQ68)', clip, ncol=5)
    _save('ratio', m, **arrays)


def exp_limitA():
    p = _dir('limitA')
    dn, r = forge_proposed(3, 'forge3')
    fcd, fx = B.fcdensenet(r['BP'], tag='forge3', model_dir=MOD)
    A.compare_figure([('FCDenseNet (re-impl.)', fcd), ('Proposed', r['den'])], dn, DT_FORGE,
                     f'{p}/fig18_limitA_eq3.png', 'Limitation A: signal leakage (EQ3, UTC190423213209, M0.91)', fk=False)
    m = {'event': D.FORGE_EVENTS['limitA'], 'proposed': A.leakage_metrics(dn, r['den']), 'fcdensenet_reimpl': A.leakage_metrics(dn, fcd)}
    _save('limitA', m, noisy=dn, proposed=r['den'], fcdensenet=fcd)


def exp_limitB():
    p = _dir('limitB')
    dn, r = forge_proposed(45, 'forge45')
    somf = D.load_forge(45, kind='somf')
    A.compare_figure([('Finest CWT scale (guide)', r['CWT'] / np.abs(r['CWT']).max() * np.percentile(np.abs(dn), 99)),
                      ('Proposed', r['den']), ('SOMF (Chen et al. 2023a)', somf)], dn, DT_FORGE,
                     f'{p}/fig19_limitB_eq45.png', 'Limitation B: residual vertical noise (EQ45*, inferred UTC190428180038)', fk=False)
    m = {'event': D.FORGE_EVENTS['limitB'], 'proposed': A.leakage_metrics(dn, r['den']), 'somf_published': A.leakage_metrics(dn, somf)}
    _save('limitB', m, noisy=dn, proposed=r['den'], somf=somf)


# ------------------------------------------------------------------------------------------- SAFOD
def exp_safod(name, eqno, trace):
    p = _dir(name)
    dn = D.load_safod(eqno)
    mrr = D.load_safod(eqno, kind='mrr')
    r = run_proposed(dn, DT_SAFOD, 0, 20, f'safod{eqno}', model_dir=MOD, large=True)
    panels = [('MRR (Chen et al. 2023b)', mrr), ('Proposed', r['den'])]
    z = _zoom(r['den'], DT_SAFOD, frac_t=0.15)
    A.compare_figure(panels, dn, DT_SAFOD, f'{p}/fig_{name}_safod_eq{eqno}.png',
                     f'SAFOD EQ{eqno} ({D.SAFOD_EVENTS[eqno][:-4]})', zoom=z, fmax=40)
    A.trace_figure([('Noisy', dn[:, trace]), ('MRR', mrr[:, trace]), ('Proposed', r['den'][:, trace])], DT_SAFOD,
                   f'{p}/fig_{name}_trace{trace}.png', f'SAFOD EQ{eqno}: trace {trace} and scalograms')
    m = {'event': D.SAFOD_EVENTS[eqno], 'zoom_window': z, 'band_pass_hz': [0, 20],
         'proposed': {**A.leakage_metrics(dn, r['den']), **_hist_summary(r['extras']['history']),
                      'n_patches': r['extras']['n_patches'], 'fit_seconds': r['extras']['fit_seconds'],
                      't_total_seconds': r['extras']['t_total']},
         'mrr_published': A.leakage_metrics(dn, mrr)}
    _save(name, m, noisy=dn, proposed=r['den'], mrr=mrr)


EXPS = {'ex1': exp_ex1, 'ex2': exp_ex2, 'ratio': exp_ratio, 'limitA': exp_limitA, 'limitB': exp_limitB,
        'ex3': lambda: exp_safod('ex3', 3, 70), 'ex4': lambda: exp_safod('ex4', 18, 120)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', nargs='+', default=['all'])
    a = ap.parse_args()
    todo = list(EXPS) if a.exp == ['all'] else a.exp
    for e in todo:
        t = time.time(); print(f'\n========== {e} ==========', flush=True)
        EXPS[e]()
        print(f'========== {e} done in {time.time() - t:.0f} s ==========', flush=True)
