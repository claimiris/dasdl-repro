"""
Figures (replicating the layout of the paper's Figs. 3-19) and an EXTENSION: quantitative signal-leakage
metrics, which the paper does not report (its evaluation is visual only).

Local similarity (Chen & Fomel, 2015, Geophysics 80(6) WD1) between the denoised section and the removed
noise is the standard leakage measure in this literature: high local similarity => coherent signal was
removed with the noise.  We use the shaping-regularised form approximated by triangle smoothing:
    c1 = S[a*b] / (S[b*b] + eps),  c2 = S[a*b] / (S[a*a] + eps),  c = sign * sqrt(|c1*c2|)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


# ----------------------------------------------------------------------------------------- metrics
def _tri(x, r):
    # triangle smoothing = two passes of a box filter, as in Madagascar's shaping regularisation
    for _ in range(2):
        x = uniform_filter(x, size=r, mode='nearest')
    return x


def local_similarity(a, b, rect=(20, 20), eps_frac=1e-3):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = _tri(a * b, rect); aa = _tri(a * a, rect); bb = _tri(b * b, rect)
    ea = eps_frac * aa.mean() + 1e-30; eb = eps_frac * bb.mean() + 1e-30
    c = (ab / (bb + eb)) * (ab / (aa + ea))
    return np.sign(ab) * np.sqrt(np.abs(c))


def leakage_metrics(noisy, den, rect=(20, 20)):
    noise = noisy - den
    ls = local_similarity(den, noise, rect)
    gc = float(np.sum(den * noise) / np.sqrt(np.sum(den ** 2) * np.sum(noise ** 2) + 1e-30))
    return dict(mean_abs_local_similarity=float(np.mean(np.abs(ls))),
                p99_abs_local_similarity=float(np.percentile(np.abs(ls), 99)),
                global_correlation_signal_noise=gc,
                energy_ratio_removed=float(np.sum(noise ** 2) / (np.sum(noisy ** 2) + 1e-30)))


# ----------------------------------------------------------------------------------------- plotting
def _cmap():
    try:
        from Utils.cmap import cmaps
        return cmaps['bluorange']
    except Exception:
        return 'seismic'


def section(ax, d, title, dt, clip=None, extent_x=None, sym=False):
    from matplotlib.colors import SymLogNorm
    nt, nx = d.shape
    ext = [0, nx if extent_x is None else extent_x, nt * dt, 0]
    if sym:   # upstream notebook display for FORGE
        im = ax.imshow(d, cmap=_cmap(), aspect='auto', extent=ext,
                       norm=SymLogNorm(linthresh=1e-4, linscale=0.5, vmin=-1e4, vmax=1e4, base=10))
    else:
        im = ax.imshow(d, cmap=_cmap(), aspect='auto', extent=ext, vmin=-clip, vmax=clip)
    ax.set_title(title, fontsize=10); ax.set_xlabel('Channel'); ax.set_ylabel('Time (s)')
    return im


def fk_spectrum(d, dt, dx=1.0):
    nt, nx = d.shape
    F = np.fft.fftshift(np.fft.fft2(d), axes=1)
    F = np.abs(F[: nt // 2 + 1])
    f = np.fft.rfftfreq(nt, dt)
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    return F, f, k


def compare_figure(panels, dn, dt, path, title, clip_pct=99.0, zoom=None, sym=False, fk=True, fmax=None):
    """panels: list of (name, denoised).  Row 1: noisy + denoised; row 2: removed noise; optional zoom + f-k rows."""
    clip = np.percentile(np.abs(dn), clip_pct)
    n = len(panels) + 1
    rows = 2 + (1 if zoom else 0) + (1 if fk else 0)
    fig, axs = plt.subplots(rows, n, figsize=(3.6 * n, 3.2 * rows), squeeze=False)
    section(axs[0, 0], dn, 'Noisy', dt, clip, sym=sym)
    for j, (name, d) in enumerate(panels, 1):
        section(axs[0, j], d, name, dt, clip, sym=sym)
        section(axs[1, j], dn - d, f'{name}: removed noise', dt, clip, sym=sym)
    axs[1, 0].axis('off')
    r = 2
    if zoom:
        (t0, t1), (x0, x1) = zoom
        i0, i1 = int(t0 / dt), int(t1 / dt)
        for j, (name, d) in enumerate([('Noisy', dn)] + list(panels)):
            sub = d[i0:i1, x0:x1]
            axs[r, j].imshow(sub, cmap=_cmap(), aspect='auto', vmin=-clip, vmax=clip, extent=[x0, x1, t1, t0])
            axs[r, j].set_title(f'{name} (zoom)', fontsize=10)
        r += 1
    if fk:
        for j, (name, d) in enumerate([('Noisy', dn)] + list(panels)):
            F, f, k = fk_spectrum(d, dt)
            axs[r, j].imshow(F, aspect='auto', cmap='jet', extent=[k[0], k[-1], f[-1], f[0]],
                             vmax=np.percentile(F, 99.5))
            if fmax: axs[r, j].set_ylim(fmax, 0)
            axs[r, j].set_title(f'{name}: f-k', fontsize=10); axs[r, j].set_xlabel('Wavenumber (1/channel)')
            axs[r, j].set_ylabel('Frequency (Hz)')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def morlet_scalogram(x, dt, freqs, w0=6.0):
    """|CWT| of a trace with a Morlet wavelet (FFT implementation; scipy.signal.cwt was removed in SciPy 1.15)."""
    n = len(x); X = np.fft.fft(x); om = 2 * np.pi * np.fft.fftfreq(n, dt)
    out = np.empty((len(freqs), n))
    for i, f in enumerate(freqs):
        s = w0 / (2 * np.pi * f)
        psi = np.pi ** -0.25 * np.exp(-0.5 * (s * om - w0) ** 2) * (om > 0) * np.sqrt(2 * np.pi * s / dt)
        out[i] = np.abs(np.fft.ifft(X * psi))
    return out


def trace_figure(traces, dt, path, title, fmax=40):
    """traces: list of (name, 1-D array).  Paper Figs. 11 and 14: traces (top) and scalograms (bottom)."""
    n = len(traces); t = np.arange(len(traces[0][1])) * dt
    freqs = np.linspace(0.5, fmax, 80)
    fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 6), squeeze=False)
    lim = np.max(np.abs(traces[0][1]))
    for j, (name, x) in enumerate(traces):
        axs[0, j].plot(t, x, 'k', lw=0.5); axs[0, j].set_ylim(-lim, lim); axs[0, j].set_title(name)
        axs[0, j].set_xlabel('Time (s)')
        S = morlet_scalogram(x, dt, freqs)
        axs[1, j].imshow(S, aspect='auto', origin='lower', cmap='jet', extent=[t[0], t[-1], freqs[0], freqs[-1]])
        axs[1, j].set_xlabel('Time (s)'); axs[1, j].set_ylabel('Frequency (Hz)')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def grid_figure(items, dt, path, title, clip, ncol):
    """Generic grid of sections, items = list of (name, 2-D array)."""
    nrow = int(np.ceil(len(items) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axs.flat: ax.axis('off')
    for ax, (name, d) in zip(axs.flat, items):
        ax.axis('on'); section(ax, d, name, dt, clip)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def weights_figure(W, path, title, n=8, seed=0):
    """Paper Fig. 17: random neurons' weight vectors reshaped to 48x48. W rows = rows of the figure."""
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(len(W), n, figsize=(1.6 * n, 1.8 * len(W)), squeeze=False)
    for r, (name, M, axis) in enumerate(W):
        idx = rng.choice(M.shape[axis], n, replace=False)
        for c, i in enumerate(idx):
            v = M[:, i] if axis == 1 else M[i, :]
            axs[r, c].imshow(v.reshape(48, 48), cmap='jet'); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(name, fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
