"""
Benchmark methods used in the paper.

SOMF  (Chen et al., 2023a, SRL 94(1)) - band-pass + structure-oriented median filter + f-k dip filter.
      Default: the authors' published result (dasdenoising-dataonly/mat_bpsomffk).  `somf_recompute`
      re-runs it with pyseistr using the parameters of pyseistr/demos/test_pyseistr_das.py and
      dasdenoising-dataonly/test_oneevent.m.

MRR   (Chen et al., 2023b, Geophysics 88(6)) - band-pass 0-20 Hz + median filter + moving rank reduction.
      Default: the authors' published result (dasmrrcoh-dataonly/processed/eqN.mat).  `mrr_recompute` re-runs
      it with pydrr.drr3d_win using the parameters of dasmrrcoh-dataonly/test_fig9.m.

FCDenseNet (Yang et al., 2023, Geophysics 88(4), "Denoising distributed acoustic sensing data using unsupervised
      deep learning").  NO public code exists and the article is pay-walled; the paper under reproduction gives no
      architecture or training details.  This is a RE-IMPLEMENTATION from the published abstract-level
      description: a fully connected network on flattened patches whose encoder and decoder are made of dense
      blocks (each FC block receives the concatenation of all previous blocks' outputs) with skip connections
      between encoder and decoder, trained without labels to reconstruct the band-passed noisy patches.
      Its results are therefore only indicative of that family of methods, not of the original implementation.
"""
import time
import numpy as np


# ------------------------------------------------------------------------------------------- SOMF
def somf_recompute(dn, dt=0.0005):
    import pyseistr as ps
    d1 = ps.bandpassc(dn, dt, 0, 200, 6, 6, 0, 0)
    pp = ps.dip2dc(d1, 2, 10, 2, 0.01, 1, 0.000001, [50, 50, 1], 1)
    d1 = ps.somf2dc(d1, pp, 8, 2, 0.01, 1)
    return d1 - ps.fkdip(d1, 0.02)


# -------------------------------------------------------------------------------------------- MRR
def mrr_recompute(dn, dt=1 / 250.0):
    """dn: (time, channel).  test_fig9.m: das_bandpass(eq',1/250,0,20); das_mf(d_bp,5,1,1); drr3d_win(...,0,50,1/250,2,4,0,512,200,1,.5,.5,.5)"""
    from Utils.bp import bandpass
    from Utils.fastops import mf_fast
    from pydrr import drr3d_win
    d_bp = bandpass(dn, dt, 0, 20, 6, 6, 0, 0)            # (time, channel)
    d_bpmf = mf_fast(d_bp.T, 5, 1, 1).T                    # MATLAB applies das_mf to the (channel, time) array
    out = drr3d_win(d_bpmf, 0, 50, dt, 2, 4, 0, 512, 200, 1, 0.5, 0.5, 0.5)
    return np.squeeze(out)


# ------------------------------------------------------------------------------------- FCDenseNet
def _build_fcdensenet(n_in, widths=(1024, 512, 256), n_fc_per_block=3):
    import tensorflow as tf
    L = tf.keras.layers

    def dense_block(x, width):
        feats = [x]
        for _ in range(n_fc_per_block):
            inp = feats[0] if len(feats) == 1 else L.Concatenate()(feats)
            feats.append(L.Dense(width, activation='relu')(inp))
        return L.Dense(width, activation='relu')(L.Concatenate()(feats[1:]))   # transition

    inp = L.Input(shape=(n_in,))
    skips, x = [], inp
    for w in widths:                       # encoder
        x = dense_block(x, w); skips.append(x)
    for w, s in zip(reversed(widths[:-1]), reversed(skips[:-1])):   # decoder with skip connections
        x = dense_block(x, w)
        x = L.Add()([x, s])
    out = L.Dense(n_in, activation='linear')(x)
    return tf.keras.Model(inp, out)


def fcdensenet(BP, w1=48, w2=48, s1=8, s2=8, epochs=100, batch_size=128, tag='fcd', model_dir='./models',
               verbose=2, large=False):
    """Unsupervised: input = target = band-passed noisy patches (same patching as the proposed method)."""
    import tensorflow as tf
    from Utils.Utils import patch_inv
    from Utils.fastops import patch_fast
    X = patch_fast(BP, w1, w2, s1, s2, dtype=np.float32 if large else None)
    model = _build_fcdensenet(w1 * w2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    ckpt = f'{model_dir}/FCDenseNet_{tag}.h5'
    cbs = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
           tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='loss', save_weights_only=True, save_best_only=True)]
    t0 = time.time()
    hist = model.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=cbs, verbose=verbose)
    fit_s = time.time() - t0
    model.load_weights(ckpt)
    out = model.predict(X, batch_size=256)
    n1, n2 = BP.shape
    den = np.array(patch_inv(out.T, n1, n2, w1, w2, s1, s2))
    tf.keras.backend.clear_session()
    return den, dict(history=hist.history, fit_seconds=fit_s, t_total=time.time() - t0, n_params=None)
