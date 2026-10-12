# Muon generator

A generative model for the muons produced by the 400 GeV proton beam: it samples complete
muons `(px, py, pz, x, y, z, pdg)` with the distribution of the simulated sample
`muons_data.h5` (2.3e9 rows, each with its own importance weight), so a downstream
simulation can use as many muons as it needs.

The generator is one normalizing flow (rational-quadratic spline couplings,
`models.MuonFlow`), saved as one checkpoint, factorized in two stages:

```
stage 1   momentum (pz, pt)    momentum part       p(pz, pt)                      3.7k params
          charge q             Bernoulli           P(mu+) = 0.529 (rate-weighted)
          azimuth phi_p        uniform [0, 2pi)    px = pt cos phi_p, py = pt sin phi_p
stage 2   position (x, y, z)   position part       p(x, y, z | pz, pt, q)          7.8k params
```

The momentum part is a flow of its own inside the model, so the momentum distribution can
be changed (e.g. tempered) without touching the position distribution at a given momentum.
The position part models the production point relative to the momentum's azimuth (the beam is
azimuthally symmetric, so only that relative angle is physical).

## Using it

```python
from generator import MuonGenerator

gen = MuonGenerator(device="cuda")        # outputs/muon_flow.pt
muons = gen.sample(1_000_000)             # (n, 7) float32: px, py, pz [GeV], x, y, z [cm], pdg
muons = gen.sample(1_000_000, momentum_temperature=1.2)   # broader momentum spectrum
log_p = gen.log_prob(muons)               # exact log density of physical muons
```

`momentum_temperature` scales the momentum part's latent (T > 1 broadens the spectrum:
at T = 1.3 the 99th percentile of |p| goes from 108 to 179 GeV); the charge fraction and
the position given the momentum are unchanged. Every generated muon has
3 < |p| < 400 GeV and pz, pt > 0 by construction. 1e6 muons take ~0.65 s on a GPU.

## Pipeline

| step | command | output |
|---|---|---|
| split the data by weight | `python split_by_weight.py 0.01` | `muons_data_split.h5` (`above_W`, `below_W`) |
| look at the data | `python see_data.py` | `outputs/figs/data_file/` |
| train the flow | `python train_flow.py --p_min 3.0` | `outputs/muon_flow.pt` |
| compare with the data | `python test_generator.py --load_hists` | `outputs/figs/*.png`, W1 and NLL |
| downstream simulation | `python simulation_test.py` | hit rate, real vs generated |

The flow trains on the whole split file: each batch mixes `above_W` (1.6e8 rows, almost
all of the rate) and `below_W` (2.2e9 rows, ~1e-4 of the rate), and the loss is the
rate-weighted NLL, the sum of the two stages' (they share no weights, so this is the
same fit as training them apart). Training took ~10 min (momentum) + ~35 min (position)
on one L40S when the stages were trained separately. `test_generator.py` caches the real side of every figure in
`outputs/real_hists.npz`: run it once without `--load_hists` to build the cache.

## Files

| file | what it is |
|---|---|
| `generator.py` | `MuonGenerator`: the two stages, `sample()` and `log_prob()` |
| `fit_space.py` | maps between physical muons and the flow's variables, with Jacobians |
| `train_flow.py` | trains the flow: p(u, v) p(\|dphi\|, asinh r, asinh z \| u, v, q) |
| `models.py` | the Neural Spline Flow (optionally conditional), `MuonFlow`, checkpoints |
| `flow_training.py` | data loading, weighted batching and the training loop |
| `test_generator.py` | real vs generated figures, W1 distances, NLL |
| `gen_eval.py` | the plots and metrics, with the real-side cache |
| `simulation_test.py` | runs real and generated muons through the transport simulation (`cuda_muons_ship`) and compares the hit rates |
| `utils.py` | reading the muon file, the beam spot |
| `see_data.py`, `split_by_weight.py`, `fit_z.py` | data inspection, the weight split, a Beta fit to z |
| `gmm/` | a Gaussian-mixture baseline for the momentum (below) |
| `compare_gmm_flow.py` | GMM vs flow on the momentum, by model size |

## The fit space (`fit_space.py`)

- **momentum** `u = logit((|p| - p_min) / (p_max - p_min))`, `v = log(pt / pz)`, with
  `p_max = 400 GeV` (the beam) and `p_min = 3 GeV`: the bounds hold by construction.
- **position** `a = |phi_p - phi_r|`, `b = asinh(r / 2 cm)`, `c = asinh((z - z0) / 20 cm)`,
  conditioned on `(u, v, q)`. The sign of `phi_p - phi_r` is independent of the charge
  in the data (P > 0 is 0.501 for both), so it is drawn at random.

Every map carries its Jacobian, so `MuonGenerator.log_prob` is the exact density in
`(px, py, pz, x, y, z)` [GeV^-3 cm^-3] times P(charge), and NLLs of different models
compare directly whatever their internal variables.

## How well it matches the data

`python test_generator.py --load_hists` (2.3e9 generated vs 2.3e9 real muons):

| | x | y | z | log10 r | \|dphi\| | px | py | pz | pt |
|---|---|---|---|---|---|---|---|---|---|
| normalized W1 | 0.0024 | 0.0010 | 0.0097 | 0.0028 | 0.0021 | 0.0048 | 0.0052 | 0.0063 | 0.0053 |

(W1 distance of each marginal divided by the real standard deviation.) The rate-weighted
NLL of 5e6 real muons is 14.6405 nats/muon, and <cos dphi> is 0.0240 generated vs
0.0250 real. Known imperfections, all at a small fraction of the rate:

- **\|dphi\| wiggles** of up to 5% around the real curve. The position part's spline box
  (+-9 sigma, 12 bins) is much wider than the data in \|dphi\| (~3.5 sigma), so only 2-3
  bins describe it; a tighter `--tail_bound` or more `--num_bins` should smooth it.
- **pt tail beyond the data**: generated pt reaches ~13 GeV at densities of 1e-7-1e-8,
  where the real sample stops near 6.5 GeV.
- **small radii**: a flat floor of ~2e-6 of the muons at r < 2 cm, where the data has
  almost none.

## How the model was chosen

All numbers are rate-weighted NLLs [nats/muon] of real muons.

- **One joint flow vs two stages.** On the last 1.7e7 `above_W` muons, a single 6D flow
  over the whole muon state (trained without them) reaches 14.640 at best, with 376k
  parameters. The two-stage generator reaches **14.639** with 11.5k parameters; it was
  trained on all the data, these rows included, which at this size makes a negligible
  difference.
- **Stage 1 size.** For the momentum alone, flows with 4 coupling layers give the same
  NLL at every width from 4 to 128 (3.6068-3.6072), so 4 x 16 was taken.
- **Stage 2 size.** Depth matters, width hardly:

  | layers \ width | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|
  | 2 | 14.6624 | 14.6612 | 14.6563 | 14.6555 |
  | 4 | 14.6416 | 14.6405 | 14.6413 | 14.6402 |
  | 6 | | **14.6387** | 14.6384 | |
  | 8 | | 14.6386 | | |

  6 x 16 (7.8k parameters) is the smallest model at the best level.
- **GMM vs flow.** A Gaussian mixture is competitive only for the smooth 2D momentum
  (K = 1024: 3.6058 vs 3.607 for the flows, and ~100x faster to sample). On the full 6D
  state it loses clearly (0.01-0.03 nats/muon behind at equal size), because of the hard
  edges (|dphi| in [0, pi], r >= 0) and the discrete charge.

## GMM baseline (`gmm/`)

`gmm_model.py` is a numpy-only mixture in the momentum part's space (JSON in, samples and
densities out); `train_gmm.py` fits it by weighted EM on the GPU over every row of the
data, `test_gmm.py` compares it with the data. `gmm/outputs/gmm_model.json` is K = 32,
fit on every row.

## Requirements

Python 3 with `torch`, `numpy`, `h5py`, `scipy`, `matplotlib`, `tqdm`;
`simulation_test.py` also needs the `cuda_muons_ship` simulation package.
