"""Summarise ../results/*/metrics.json as Markdown (pasted into the README's Results section)."""
import json, os, glob

R = '../results'
NAMES = {'proposed': 'Proposed', 'somf_published': 'SOMF', 'fcdensenet_reimpl': 'FCDenseNet (re-impl.)', 'mrr_published': 'MRR'}


def row(exp, label, m):
    return (f"| {exp} | {label} | {m['mean_abs_local_similarity']:.4f} | {m['p99_abs_local_similarity']:.3f} | "
            f"{m['global_correlation_signal_noise']:.4f} | {m['energy_ratio_removed']:.3f} |")


def main():
    out = ['| Experiment | Method | mean \\|LS\\| ↓ | p99 \\|LS\\| ↓ | corr(signal, noise) ↓ | energy removed |',
           '|---|---|---|---|---|---|']
    train = ['| Experiment | Patches (train/total) | Epochs run | Final LogCosh loss | Fit time on T4 (s) |', '|---|---|---|---|---|']
    for exp in ['ex1', 'ex2', 'ex3', 'ex4', 'limitA', 'limitB']:
        p = os.path.join(R, exp, 'metrics.json')
        if not os.path.exists(p): continue
        m = json.load(open(p))
        for k, lab in NAMES.items():
            if k in m: out.append(row(exp, lab, m[k]))
        pr = m['proposed']
        if 'epochs_run' in pr:
            train.append(f"| {exp} | {pr.get('n_patches', '')} | {pr['epochs_run']} | {pr['last_loss']:.4f} | {pr['fit_seconds']:.0f} |")
    p = os.path.join(R, 'ratio', 'metrics.json')
    if os.path.exists(p):
        m = json.load(open(p))
        for r in ['10%', '30%', '50%', '70%', '90%']:
            if r in m:
                out.append(row('ratio', f'Proposed, {r} patches', m[r]))
                train.append(f"| ratio {r} | {m[r]['n_train_patches']} | {m[r]['epochs_run']} | {m[r]['last_loss']:.4f} | {m[r]['fit_seconds']:.0f} |")
    print('\n'.join(out)); print(); print('\n'.join(train))


if __name__ == '__main__':
    main()
