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
    px,py,pz,_x,_y,z,pdg,weight = get_real_muons('/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample.h5').T
    pt = calc_pt(px, py)
    phi = calc_phi(px, py)
    x,y = generate_position(px.shape[0])

    print(f"Number of samples: {px.shape[0]}")
    print(f"Rate: {weight.sum()} muons per second")

    weights_unique = np.unique(weight)
    weights_unique = weights_unique[weights_unique > 0]  # Filter out zero weights


    ######## momentum distribution
    ax = hist2d(pz, pt, W=weight)
    ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
    ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 13)
    plt.savefig(figs_dir+'momentum.png')
    if show: plt.show()

    for w in weights_unique:
        count = np.sum(weight == w)
        print(f"Weight: {w}, Count: {count}")
        ax = hist2d(pz[weight == w], pt[weight == w])
        w = f"{w:.2f}".replace(".", "_")
        ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
        ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 13)
        plt.title(f'Weight: {w}', fontsize=fontsize)
        plt.savefig(figs_dir+f'momentum_weight_{w}.png')
        if show: plt.show()


    ##### z_dist
    hist1d(z,log = True)
    target = -2.14
    plt.axvline(target, color='red', linestyle='--', label='Target Position')
    plt.axvline(target+1.595, color='red', linestyle='--')
    plt.xlabel('z [cm]', fontsize=14)
    plt.savefig(figs_dir+'z_dist.png')
    if show: plt.show()

    ###### position_dist
    hist2d(_x, _y)
    plt.xlabel('x [cm]', fontsize=14)
    plt.ylabel('y [cm]', fontsize=14)
    circ = plt.Circle((0, 0), 15, color='r', fill=False)
    plt.gca().add_patch(circ)
    plt.savefig(figs_dir+'original_position_dist.png')
    if show: plt.show()

    hist2d(x, y)
    plt.xlabel('x [cm]', fontsize=14)
    plt.ylabel('y [cm]', fontsize=14)
    circ = plt.Circle((0, 0), 0.15, color='r', fill=False)
    plt.gca().add_patch(circ)
    plt.savefig(figs_dir+'position_dist.png')
    if show: plt.show()

    ##### phi distribution
    hist1d(phi, density=True, histtype='step', log=False)
    plt.xlabel('phi [rad]', fontsize=14)
    plt.savefig(figs_dir+'phi_dist.png')
    if show: plt.show()
