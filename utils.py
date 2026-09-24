"""Data loading and small shared helpers.

The muon file is a flat table: one dataset per column (px, py, pz, x, y, z, pdg,
weight) and one row per muon, each row carrying its own importance weight. See
input_files/unify_h5.py, which builds it.
"""
import os

import torch
import h5py
import numpy as np

COLS = ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight')
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),'muons_data.h5')
CHUNK = 5_000_000        # rows read from HDF5 at a time


def get_real_muons(file_path=DEFAULT_DATA, num_samples=None, columns=COLS,
                   rescale_weights=True):
    """The muon table as one (N, len(columns)) float32 array, columns in `columns`
    order -- (px, py, pz, x, y, z, pdg, weight) by default, i.e. the layout the rest of
    the repo indexes positionally.

    `num_samples` caps the rows read (from the head of the file, which is a
    concatenation of statistically equivalent jobs, so the head is a fair sub-sample).
    With `rescale_weights`, a capped read has its weights multiplied by N_total/n so the
    sub-sample still carries the full physical rate; without it the weights are the
    file's own. Reading is chunked straight into the output array -- a whole-column read
    of 2.3e9 rows would be 9 GB of temporaries per column.
    """
    with h5py.File(file_path, 'r') as f:
        n_total = f[columns[0]].shape[0]
        n = n_total if not num_samples else min(num_samples, n_total)
        out = np.empty((n, len(columns)), dtype=np.float32)
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            for c, name in enumerate(columns):
                out[sl, c] = f[name][sl]
    msg = f"Loaded {n:,} of {n_total:,} muons from {file_path}"
    if 'weight' in columns:
        w = out[:, columns.index('weight')]
        if rescale_weights and n < n_total:
            w *= n_total / n
        msg += f" (sum of weights {w.sum(dtype=np.float64):,.4g})"
    print(msg)
    return out


def calc_pt(px, py):
    return np.sqrt(px**2 + py**2)

def calc_phi(px, py):
    return np.arctan2(py, px)

def generate_px_py(pt):
    phi = np.random.uniform(0, 2 * np.pi, size=pt.shape[0])
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return px, py
def generate_charge(num_samples):
    # Randomly assign charge +1 or -1 with equal probability
    charges = np.random.choice([-1, 1], size=num_samples)
    return charges
def generate_pdg(num_samples):
    # Randomly assign PDG code for muons (13 for muon, -13 for anti-muon)
    pdg_codes = (-13) * generate_charge(num_samples)
    return pdg_codes

def generate_position(num_samples, SmearBeamRadius=5.0, sigma=1.6, fixed_z=-1.9):
    #ring transformation
    gauss_x = np.random.normal(0, sigma, size=num_samples)
    gauss_y = np.random.normal(0, sigma, size=num_samples)
    uniform = np.random.uniform(0, 1, size=num_samples)
    _phi = uniform * 2 * np.pi
    x = SmearBeamRadius * np.cos(_phi) + gauss_x
    y = SmearBeamRadius * np.sin(_phi) + gauss_y
    x = x / 100
    y = y / 100
    z = np.full(num_samples, fixed_z)
    return x, y, z


def train_val_split(X, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X) * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], X[val_idx]


def standardize_train_val(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-12
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    return X_train_s, X_val_s, mean, std

if __name__ == "__main__":
    print(get_real_muons(num_samples=1000).shape)
