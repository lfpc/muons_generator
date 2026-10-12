"""Maps between physical muons and the fit space of the muon flow (models.MuonFlow).

A muon is (px, py, pz, x, y, z, pdg). The generator describes it by

    momentum  (u, v)     u = logit((|p| - p_min) / (p_max - p_min)),  v = log(pt / pz)
    position  (a, b, c)  a = |dphi|,  b = asinh(r / r_scale),  c = asinh((z - z0) / z_scale)

plus the charge q = -sign(pdg) and the momentum azimuth phi_p, which need no model;
dphi = phi_p - phi_r is the azimuth of the production point relative to the momentum's.
u keeps p_min < |p| < p_max and v keeps pz, pt > 0, by construction.

Each map to the fit space also returns the log of its Jacobian, so densities convert
exactly: log p(physical) = log p(fit) + log_jac. `m` holds the constants of the space
(p_min, p_max, r_scale, z0, z_scale, r_max); every checkpoint stores its own.
"""
import numpy as np

TWO_PI = 2.0 * np.pi
P_MAX = 400.0                 # |p| below the 400 GeV proton beam [GeV]
R_SCALE, Z_SCALE = 2.0, 20.0  # where the asinh turns from linear to log [cm]


def logit_p(p, m):
    return np.log(p - m["p_min"]) - np.log(m["p_max"] - p)


def momentum_to_fit(pz, pt, m):
    """(pz, pt) -> (u, v), and log |d(u, v) / d(pz, pt)|."""
    p = np.hypot(pz, pt)
    uv = np.column_stack([logit_p(p, m), np.log(pt / pz)])
    log_jac = (np.log(m["p_max"] - m["p_min"]) - np.log(p - m["p_min"]) - np.log(m["p_max"] - p)
               + np.log(p) - np.log(pz) - np.log(pt))
    return uv, log_jac


def fit_to_momentum(uv, m):
    """(u, v) -> (pz, pt)."""
    p = m["p_min"] + (m["p_max"] - m["p_min"]) / (1.0 + np.exp(-uv[:, 0]))
    pz = p / np.sqrt(1.0 + np.exp(2.0 * uv[:, 1]))
    return pz, pz * np.exp(uv[:, 1])


def position_to_fit(px, py, x, y, z, m):
    """(a, b, c), and the log Jacobian from them to (x, y, z) given the momentum
    azimuth: x, y enter as r dr dphi_r and dphi has a random sign (a factor 1/2)."""
    r = np.hypot(x, y)
    dphi = np.arctan2(py, px) - np.arctan2(y, x)
    abc = np.column_stack([np.abs((dphi + np.pi) % TWO_PI - np.pi),
                           np.arcsinh(r / m["r_scale"]),
                           np.arcsinh((z - m["z0"]) / m["z_scale"])])
    log_jac = (-np.log(2.0) - np.log(r)
               - 0.5 * np.log(r ** 2 + m["r_scale"] ** 2)
               - 0.5 * np.log((z - m["z0"]) ** 2 + m["z_scale"] ** 2))
    return abc, log_jac


def fit_to_position(abc, phi_p, m, rng):
    """(a, b, c) and the momentum azimuth -> (x, y, z). |dphi| is folded back into
    [0, pi] and given a random sign; r is folded to |r| (the flow can put a sample at
    asinh r < 0) and capped at r_max, the largest radius in the data."""
    dphi = np.abs(abc[:, 0]) % TWO_PI
    dphi = np.minimum(dphi, TWO_PI - dphi) * rng.choice([-1.0, 1.0], size=len(dphi))
    r = np.minimum(np.abs(m["r_scale"] * np.sinh(abc[:, 1])), m["r_max"])
    phi_r = phi_p - dphi
    return r * np.cos(phi_r), r * np.sin(phi_r), m["z0"] + m["z_scale"] * np.sinh(abc[:, 2])
