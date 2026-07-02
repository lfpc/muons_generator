import torch
import h5py
import numpy as np

def get_real_muons(file_path, num_samples=None):
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        px = f['px'][:] if num_samples is None else f['px'][:num_samples]
        py = f['py'][:] if num_samples is None else f['py'][:num_samples]
        pz = f['pz'][:] if num_samples is None else f['pz'][:num_samples]
        x_ = f['x'][:] if num_samples is None else f['x'][:num_samples]
        y_ = f['y'][:] if num_samples is None else f['y'][:num_samples]
        z_ = f['z'][:] if num_samples is None else f['z'][:num_samples]
        pdg = f['pdg'][:] if num_samples is None else f['pdg'][:num_samples]
        weight = f['weight'][:] if num_samples is None else f['weight'][:num_samples]
        data = np.stack([px, py, pz, x_, y_, z_, pdg, weight], axis=1)
    return data

def get_real_muons_per_weight(file_path, num_samples=None):
    print(f"Loading data from {file_path}...")
    data = {}
    with h5py.File(file_path, 'r') as f:
        for group_name in f.keys():
            print(f"Group: {group_name}")
            group = f[group_name]
            weight = group.attrs['weight']
            px = group['px'][:] if num_samples is None else group['px'][:num_samples]
            py = group['py'][:] if num_samples is None else group['py'][:num_samples]
            pz = group['pz'][:] if num_samples is None else group['pz'][:num_samples]
            x_ = group['x'][:] if num_samples is None else group['x'][:num_samples]
            y_ = group['y'][:] if num_samples is None else group['y'][:num_samples]
            z_ = group['z'][:] if num_samples is None else group['z'][:num_samples]
            pdg = group['pdg'][:] if num_samples is None else group['pdg'][:num_samples]
            data[weight] = np.stack([px, py, pz, x_, y_, z_, pdg], axis=1)
    return data

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

def generate_position(num_samples, SmearBeamRadius=5.0, sigma=1.6):
    #ring transformation
    gauss_x = np.random.normal(0, sigma, size=num_samples)
    gauss_y = np.random.normal(0, sigma, size=num_samples)
    uniform = np.random.uniform(0, 1, size=num_samples)
    _phi = uniform * 2 * np.pi
    x = SmearBeamRadius * np.cos(_phi) + gauss_x
    y = SmearBeamRadius * np.sin(_phi) + gauss_y
    x = x / 100
    y = y / 100
    #z = np.zeros(num_samples) 
    return x, y#, z


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
    file_path = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample.h5'
    data = get_real_muons(file_path)
    print(data.shape)

