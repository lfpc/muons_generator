"""Data loading and small shared helpers.

The muon file is a flat table: one dataset per column (px, py, pz, x, y, z, pdg,
weight) and one row per muon, each row carrying its own importance weight.
split_by_weight.py splits it into the groups above_W and below_W.
"""
import os

import h5py
import numpy as np

COLS = ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight')
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),'muons_data.h5')
CHUNK = 5_000_000        # rows read from HDF5 at a time


def get_real_muons(file_path=DEFAULT_DATA, num_samples=None, columns=COLS,
                   rescale_weights=True, group=None):
    """The muon table as one (N, len(columns)) float32 array, columns in `columns`
    order -- (px, py, pz, x, y, z, pdg, weight) by default, i.e. the layout the rest of
    the repo indexes positionally.

    `num_samples` caps the rows read (from the head of the file, which is a
    concatenation of statistically equivalent jobs, so the head is a fair sub-sample).
    With `rescale_weights`, a capped read has its weights multiplied by N_total/n so the
    sub-sample still carries the full physical rate; without it the weights are the
    file's own. Reading is chunked straight into the output array -- a whole-column read
    of 2.3e9 rows would be 9 GB of temporaries per column.

    `group` reads a single group of a split file (split_by_weight.py), e.g. 'above_W';
    the other groups are never touched and n_total is that group's size.
    """
    with h5py.File(file_path, 'r') as f:
        if group is not None:
            f = f[group]
        n_total = f[columns[0]].shape[0]
        n = n_total if not num_samples else min(num_samples, n_total)
        out = np.empty((n, len(columns)), dtype=np.float32)
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            for c, name in enumerate(columns):
                out[sl, c] = f[name][sl]
    msg = f"Loaded {n:,} of {n_total:,} muons from {file_path}"
    if group is not None:
        msg += f" [{group}]"
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


def generate_position(num_samples, SmearBeamRadius=5.0, sigma=1.6, fixed_z=-1.9):
    """The beam spot: a ring of radius SmearBeamRadius [cm] smeared by a Gaussian of
    width sigma [cm], returned in m, at a fixed z. simulation_test.py adds it to the
    production point of real and generated muons alike."""
    phi = np.random.uniform(0, 2 * np.pi, size=num_samples)
    x = SmearBeamRadius * np.cos(phi) + np.random.normal(0, sigma, size=num_samples)
    y = SmearBeamRadius * np.sin(phi) + np.random.normal(0, sigma, size=num_samples)
    return x / 100, y / 100, np.full(num_samples, fixed_z)
