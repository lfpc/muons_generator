import numpy as np
from utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

figs_dir = 'outputs/figs/'
show = False

fontsize = 14
def hist1d(x, W = None, log=False, density=False, histtype='bar'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    h1, bins1, im1 = ax.hist(x, bins=100, color='blue', alpha=0.7, histtype=histtype, label='Data', density=density, weights=W, log=log)
    ax.set_ylabel('Counts', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)

def hist2d(x,y, W = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # First histogram (Enriched Sample)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    h1, xedges, yedges, im1 = ax.hist2d(x,y, bins=200, cmap='viridis', norm=LogNorm(), density=False, weights=W)
    cbar1 = fig.colorbar(im1, ax=ax)
    return ax

if __name__ == "__main__":
    data = get_real_muons_per_weight('/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5')

    for weight, arr in data.items():
        px, py, pz = arr[:, 0], arr[:, 1], arr[:, 2]
        print(f"Weight: {weight}, N: {px.shape[0]}, Rate: {weight * px.shape[0]:.2f} muons/s")

    # concatenate all weights for full-sample plots
    all_arr = np.concatenate(list(data.values()), axis=0)
    all_W   = np.concatenate([np.full(arr.shape[0], w) for w, arr in data.items()])
    all_px, all_py, all_pz = all_arr[:, 0], all_arr[:, 1], all_arr[:, 2]
    all_x, all_y, all_z   = all_arr[:, 3], all_arr[:, 4], all_arr[:, 5]
    all_pt  = calc_pt(all_px, all_py)
    all_phi = calc_phi(all_px, all_py)
    gen_x, gen_y = generate_position(all_arr.shape[0])

    ######## momentum distribution
    ax = hist2d(all_pz, all_pt, W=all_W)
    ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
    ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 13)
    plt.savefig(figs_dir + 'momentum.png')
    if show: plt.show()

    for weight, arr in data.items():
        px, py, pz = arr[:, 0], arr[:, 1], arr[:, 2]
        pt = calc_pt(px, py)
        ax = hist2d(pz, pt)
        w_str = f"{weight:.2f}".replace(".", "_")
        ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
        ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 13)
        plt.title(f'Weight: {weight:.2f}', fontsize=fontsize)
        plt.savefig(figs_dir + f'momentum_weight_{w_str}.png')
        if show: plt.show()

    ##### z_dist
    hist1d(all_z, W=all_W, log=True)
    target = -2.14
    plt.axvline(target, color='red', linestyle='--', label='Target Position')
    plt.axvline(target + 1.595, color='red', linestyle='--')
    plt.xlabel('z [cm]', fontsize=14)
    plt.savefig(figs_dir + 'z_dist.png')
    if show: plt.show()

    ###### position_dist
    hist2d(all_x, all_y, W=all_W)
    plt.xlabel('x [cm]', fontsize=14)
    plt.ylabel('y [cm]', fontsize=14)
    circ = plt.Circle((0, 0), 15, color='r', fill=False)
    plt.gca().add_patch(circ)
    plt.savefig(figs_dir + 'original_position_dist.png')
    if show: plt.show()

    hist2d(gen_x, gen_y)
    plt.xlabel('x [cm]', fontsize=14)
    plt.ylabel('y [cm]', fontsize=14)
    circ = plt.Circle((0, 0), 0.15, color='r', fill=False)
    plt.gca().add_patch(circ)
    plt.savefig(figs_dir + 'position_dist.png')
    if show: plt.show()

    ##### phi distribution
    hist1d(all_phi, W=all_W, density=True, histtype='step', log=False)
    plt.xlabel('phi [rad]', fontsize=14)
    plt.savefig(figs_dir + 'phi_dist.png')
    if show: plt.show()
