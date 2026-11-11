"""Per-layer geometry of hidden states: dimension, cone, spectrum.

Hidden states live in a space with a shape of its own. Some layers
squeeze their states into a thin cone; others spread them across
many directions. GEOMETRY-SCAN measures that shape per layer:
intrinsic dimension, anisotropy cone strength, and how much of
the variance the first principal direction owns. Pure NumPy.
"""

import numpy as np


def _checked(states):
    s = np.asarray(states, dtype=float)
    if s.ndim != 2:
        raise ValueError("states must have shape (N, D)")
    if s.shape[0] < 2:
        raise ValueError("states must contain at least two samples")
    return s


def intrinsic_dimension(states):
    """Participation ratio of covariance eigenvalues."""
    s = _checked(states)
    xc = s - s.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / s.shape[0]
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(np.sum(lam))
    if total < 1e-12:
        return 0.0
    return float(np.sum(lam) ** 2 / np.sum(lam ** 2))


def anisotropy(states):
    """Mean pairwise cosine between raw state vectors (cone strength)."""
    s = _checked(states)
    norms = np.clip(np.linalg.norm(s, axis=1, keepdims=True), 1e-12, None)
    unit = s / norms
    n = s.shape[0]
    sim = unit @ unit.T
    off = sim[~np.eye(n, dtype=bool)]
    return float(np.mean(off))


def layer_geometry(states, name=None):
    """One geometry entry for one layer."""
    s = _checked(states)
    xc = s - s.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / s.shape[0]
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(np.sum(lam))
    pc1 = float(lam[-1]) / total if total > 0 else 0.0
    return {
        "name": name,
        "samples": int(s.shape[0]),
        "dim": int(s.shape[1]),
        "intrinsic_dimension": intrinsic_dimension(s),
        "pc1_variance_share": pc1,
        "anisotropy": anisotropy(s),
        "mean_norm": float(np.mean(np.linalg.norm(s, axis=1))),
    }


def geometry_report(states_by_layer, names=None):
    """Full per-layer geometry scan."""
    if not isinstance(states_by_layer, (list, tuple)) or len(states_by_layer) == 0:
        raise ValueError("states_by_layer must be a non-empty list")
    if names is not None and len(names) != len(states_by_layer):
        raise ValueError("names must align with states_by_layer")
    entries = [layer_geometry(s, name=names[i] if names else None)
               for i, s in enumerate(states_by_layer)]
    ids = [e["intrinsic_dimension"] for e in entries]
    anisos = [e["anisotropy"] for e in entries]
    return {
        "num_layers": len(entries),
        "layers": entries,
        "lowest_dimension_layer": int(np.argmin(ids)),
        "highest_anisotropy_layer": int(np.argmax(anisos)),
    }