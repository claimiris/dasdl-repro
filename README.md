# dasdl-repro

Reproduction and extension of

> Saad, O. M., Ravasi, M., & Alkhalifah, T. (2024). **Noise attenuation in distributed acoustic sensing data using a guided unsupervised deep learning network.** *Geophysics*, 89(6). (arXiv: [2405.07660](https://arxiv.org/abs/2405.07660), preprint title "Signal Enhancement in DAS Data Using a Guided Unsupervised Deep Learning Network")

built on the authors' official code, [DeepWave-KAUST/DLDAS_Denoising-pub](https://github.com/DeepWave-KAUST/DLDAS_Denoising-pub) (commit `0d59ae4`, MIT).
The upstream repository only covers one FORGE event (EQ68). This repo runs **every field example in the paper** (5 FORGE events + 2 SAFOD events), its **benchmarks** (SOMF, FCDenseNet, MRR), the **training-ratio**, **block-output**, **weight** and **timing** studies, and adds **quantitative signal-leakage metrics**, which the paper does not report.

**Run it:** open [`notebooks/Reproduce_All_Colab.ipynb`](notebooks/Reproduce_All_Colab.ipynb) in Google Colab with a T4 GPU and run all cells, or locally:

```bash
cd notebooks
python -m tests.test_fastops                 # checks the fast helpers against the original code
python run_experiments.py --exp all          # or: --exp ex1 ex2 ratio limitA limitB ex3 ex4
```

On TensorFlow ≥ 2.16 set `TF_USE_LEGACY_KERAS=1` first (see Edit E1). Data are downloaded automatically.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/DASDL_Main.ipynb`, `notebooks/Utils/`, `notebooks/py_cwt2d/`, `Matlab_CWT_Version/` | Upstream code (with the edits listed below) |
| `notebooks/Utils/fastops.py` | Vectorised, output-identical versions of `patch`, `mf` and the single-scale CWT (needed for the 800 × 14,999 SAFOD records) |
| `notebooks/repro/data.py` | Downloads and loads FORGE/SAFOD events and the published benchmark outputs |
| `notebooks/repro/pipeline.py` | The upstream notebook's workflow as a function (BP → MF → CWT → DL → f-k dip filter) |
| `notebooks/repro/benchmarks.py` | SOMF, MRR (recompute options) and the FCDenseNet re-implementation |
| `notebooks/repro/analysis.py` | Figure layouts of the paper and leakage metrics |
| `notebooks/run_experiments.py` | One entry point per experiment |
| `notebooks/tests/test_fastops.py` | Equivalence tests |
| `results/<exp>/` | Figures (`.png`) and `metrics.json` from the Colab T4 run |

## Data

The paper cites Lellouch et al. (2021) for both datasets and gives no download link. The exact windows it uses are published (GPL-3) by Y. Chen, author of the benchmark papers:

* **FORGE** (well 78-32, 2019, 1 m channel spacing, 10 m gauge, dt = 0.5 ms, 2000 samples × 960 channels starting 50 samples before the P arrival): [chenyk1990/dasdenoising-dataonly](https://github.com/chenyk1990/dasdenoising-dataonly), `mat_raw/eq-N.mat`. The upstream `data/eq-68.mat` is identical to `mat_raw/eq-68.mat`, and the tests check this.
* **SAFOD** (2017, 1 m spacing, dt = 4 ms, 800 channels × 14,999 samples): [chenyk1990/dasmrrcoh-dataonly](https://github.com/chenyk1990/dasmrrcoh-dataonly), `raw/<UTC>_mag<M>.mat`. Event numbers follow `processed/EQNO2RawFileName.txt`.

| Paper example | Paper file / event | Data used here | Mapping |
|---|---|---|---|
| Ex. 1 (Figs 3–5) | FORGE `UTC190502024438`, ML 0.065, "EQ68" | FORGE `eq-68` | verified (figure label, upstream repo) |
| Ex. 2 (Figs 6–8) | FORGE `UTC190428175708`, ML −0.087, "EQ44" | FORGE `eq-44` | verified (figure label) |
| Ex. 3 (Figs 9–11) | SAFOD `2017-06-23T22:03:22.38_mag2.17`, "EQ3" | SAFOD EQ3 | verified (file name) |
| Ex. 4 (Figs 12–14) | SAFOD `2017-07-02T03:17:32.30_mag2.86`, "EQ18" | SAFOD EQ18 | verified (file name) |
| Limitation A (Fig 18) | FORGE `UTC190423213209`, M 0.91 | FORGE `eq-3` | verified (`test_oneevent.m`: "ii=3: …UTC190423213209.sgy") |
| Limitation B (Fig 19) | FORGE `UTC190428180038`, M 1.07 | FORGE `eq-45` | **inferred**: no public index; the catalogue is chronological and this file is 3.5 min after EQ44 |
| Block outputs (Fig 15) | FORGE `UTC190426080738`, ML −0.957 | FORGE `eq-68` | **substituted**: no public index for this file |
| Training ratio (Fig 16), weights (Fig 17), timing (Sec 4.4) | "first field data" | FORGE `eq-68` | as in the paper |

## Benchmarks

| Method | Paper reference | What is used here |
|---|---|---|
| SOMF (BP + structure-oriented median filter + f-k) | Chen et al., 2023a, SRL 94(1) | The authors' published output, `dasdenoising-dataonly/mat_bpsomffk/eq-N.mat`. `benchmarks.somf_recompute` re-runs it with `pyseistr`, using the parameters from the authors' demo. |
| MRR (BP 0–20 Hz + MF + moving rank reduction) | Chen et al., 2023b, Geophysics 88(6) | The authors' published output, `dasmrrcoh-dataonly/processed/eqN.mat`. `benchmarks.mrr_recompute` re-runs it with `pydrr.drr3d_win`, using the parameters of `test_fig9.m`. |
| FCDenseNet (unsupervised) | Yang et al., 2023, Geophysics 88(4) | **Re-implementation.** No public code exists, the article is pay-walled, and the reproduced paper gives no settings. This version is a fully connected network on 48 × 48 patches with dense blocks (each FC layer sees all previous outputs) and encoder–decoder skip connections. It is trained without labels to reconstruct band-passed patches (MSE, Adam 1e-3, ≤100 epochs, early stopping 5). Treat its results as indicative only. |

## Settings of the proposed method

The settings are exactly those of the upstream notebook: band-pass (FORGE 1e-3–250 Hz), 5-point median filter along time, 19 Mexican-hat CWT scales `arange(1, 10.5, 0.5)` with the finest (19th) as the guide, 48 × 48 patches with an 8-sample shift, D1 = 1024 (encoder 1024/512/256, decoder 512/1024/1024), LogCosh loss, Adam, batch 128, 100 epochs, early stopping (patience 5) and the upstream learning-rate schedule, followed by the f-k dip filter (w = 0.02).

The paper does not give the SAFOD band-pass. **0–20 Hz** is used, the same band as the MRR benchmark, so both methods start from the same band.

## Edits to the original code

| # | File | Change | Why |
|---|---|---|---|
| E1 | `notebooks/Utils/DLModel.py` | `from keras.layers import Lambda` → `from tensorflow.keras.layers import Lambda`; `from keras import backend as K` → `from tensorflow.keras import backend as K`; run with `TF_USE_LEGACY_KERAS=1` | On TF ≥ 2.16, `keras` is Keras 3 while `tensorflow.keras` (legacy) is Keras 2, and mixing them breaks the model. |
| E2 | `notebooks/Utils/fk.py:62` | `dtype=np.complex_` → `dtype=np.complex128` | `np.complex_` was removed in NumPy 2.0. The dtype is identical. |
| E3 | `notebooks/Utils/DLModel.py` `Train()` | New **optional** keyword arguments: `train_ratio`, `patch_fn`, `dtype`, `seed`, `model_dir`, `verbose`, `return_extras` | Needed for the training-ratio study, the SAFOD memory footprint, timing and the weight figure. With the defaults, `Train()` runs the original computation. |
| A1 | `notebooks/Utils/fastops.py` (new) | Vectorised `patch_fast`, `mf_fast`, `cwt_scale` | The original Python loops take about 5 min per FORGE event and run out of memory on SAFOD (about 180k float64 patches, built from a list). They produce identical output; see `tests/test_fastops.py`. |

No model, loss, hyper-parameter or algorithm was changed.

## Extension: quantitative leakage metrics

The paper's evaluation is only visual. For every method and event, `metrics.json` reports:
* the mean and 99th-percentile |local similarity| between the denoised section and the removed noise (Chen & Fomel, 2015; triangle-smoothing approximation of the shaping-regularised form, 20 × 20 radius). Lower values mean less signal leakage;
* the global correlation between the denoised data and the removed noise;
* the fraction of energy removed.

## Results

Results from the Colab T4 run are added below once the experiments finish.

## Not reproducible from public material

* FCDenseNet: no code and no settings (re-implemented, see above). The paper also says FCDenseNet "did not converge" on Ex. 2, which a re-implementation cannot confirm.
* Fig. 15 (`UTC190426080738`) and the zoom rectangles of Figs 4, 7, 10, 13: not identifiable. EQ68 and automatically chosen windows are used instead.
* Exact timings: the paper used an RTX 2080 Ti, and this run used a Colab T4. Batch size is not stated in the paper, so the upstream value of 128 is used.

## Licences and credit

* Upstream code: MIT, © 2024 DeepWave-KAUST (see `LICENSE`). `Utils/mf.py`, `bp.py` and `fk.py` are GPL-3 ports by Y. Chen, as stated in their headers.
* The data are not redistributed here; they are downloaded at run time from the GPL-3 repositories above.
* Please cite the original paper, Lellouch et al. (2021) for the data, and Chen et al. (2023a, 2023b) and Yang et al. (2023) for the benchmarks.
