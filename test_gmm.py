import numpy as np
import pickle
from train_gmm import sample_gmm
from see_data import hist2d
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser(description="Train GMM on muon data")
parser.add_argument("--num_samples", type=int, default=100_000, help="Number of samples to use for training")
args = parser.parse_args()


file_path = 'outputs/gmm_model.pkl'
figs_dir = 'outputs/figs/'
show = True
with open(file_path, 'rb') as f:
    gmm = pickle.load(f)
    
mean = gmm['mean']
std = gmm['std']
gmm = gmm['gmm']
pz_gen, pt_gen, comp = sample_gmm(gmm, mean, std, n_samples=args.n_samples, seed=0)


######## momentum distribution
fontsize = 14
ax = hist2d(pz_gen, pt_gen)
ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
ax.set_xlim(0, 400)
ax.set_ylim(0, 13)
plt.savefig(figs_dir+'generated_momentum.png')
if show: plt.show()
