"""Node-deployment generators.

Each generator is a pure function that maps ``(num_nodes, area, rng)`` to an
``(num_nodes, 2)`` float array of node positions, with every coordinate in
``[-area, area]`` on each axis (the base station sits at the origin). Generation
is decoupled from both the physics layer (``model.py``) and the runner
(``main.py``): a new scenario is one function plus one ``REGISTRY`` entry.

Scenario-specific knobs are module-level constants with sensible defaults (edit
in code when a non-default value is needed) — the runner only exposes the
universal ``--deployment``/``--num-nodes``/``--seed`` flags.

A deployment is fully reproducible from ``(deployment, num_nodes, seed)``: the
caller owns a single ``numpy`` ``Generator`` and threads it through positions
*and* per-node ``Vpre``.
"""

import math

import numpy as np
from scipy.stats import qmc

# ---------------------------------------------------------------------------- #
#  Scenario knobs (sensible defaults; edit here for a non-default layout)
# ---------------------------------------------------------------------------- #
POISSON_RADIUS = 30          # min-distance radius for Poisson-disk sampling
GRID_JITTER = 0.0            # uniform +/- jitter added per grid point (0 = pure lattice)
GAUSS_BLOBS = 4              # number of Gaussian clusters
GAUSS_STD_FRAC = 1 / 6       # blob std-dev as a fraction of area
EDGE_EXP = 2.0               # edge-bias steepness (higher = denser toward edges)


def poisson_disk(num_nodes, area, rng):
    """Poisson-disk (blue-noise) sampling — even spread with a min spacing.

    Samples over ``[0, 2*area]^2``; if the disk sampler places fewer than
    ``num_nodes`` points (it stops when no candidate respects ``radius``), the
    shortfall is topped up with uniform-random points so exactly ``num_nodes``
    are returned. Coordinates are then shifted to ``[-area, area]^2``.
    """
    engine = qmc.PoissonDisk(d=2, radius=POISSON_RADIUS, rng=rng,
                             ncandidates=num_nodes,
                             l_bounds=0, u_bounds=area * 2)
    sample = engine.random(num_nodes)

    shortfall = num_nodes - len(sample)
    if shortfall > 0:
        extra = rng.uniform(0, area * 2, size=(shortfall, 2))
        sample = np.append(sample, extra, axis=0)

    return sample - area


def uniform(num_nodes, area, rng):
    """Uniform i.i.d. random placement — the classic WSN baseline."""
    return rng.uniform(-area, area, size=(num_nodes, 2))


def grid(num_nodes, area, rng):
    """Regular square lattice (optionally jittered) — a controlled reference.

    Places ``ceil(sqrt(num_nodes))`` points per axis evenly across the area and
    keeps the first ``num_nodes`` (a non-square count leaves a partial boundary
    row). ``GRID_JITTER`` perturbs each point; the result is clipped to bounds.
    """
    side = math.ceil(math.sqrt(num_nodes))
    axis = np.linspace(-area, area, side)
    xx, yy = np.meshgrid(axis, axis)
    pts = np.column_stack([xx.ravel(), yy.ravel()])[:num_nodes]

    if GRID_JITTER > 0:
        pts = pts + rng.uniform(-GRID_JITTER, GRID_JITTER, size=pts.shape)
        pts = np.clip(pts, -area, area)
    return pts


def gaussian_clusters(num_nodes, area, rng):
    """Gaussian blobs — models hotspot / clustered deployments.

    Draws ``GAUSS_BLOBS`` centers uniformly, assigns nodes round-robin to blobs,
    samples each from an isotropic Gaussian of std ``area * GAUSS_STD_FRAC``, and
    clips to bounds.
    """
    centers = rng.uniform(-area, area, size=(GAUSS_BLOBS, 2))
    assign = np.arange(num_nodes) % GAUSS_BLOBS
    std = area * GAUSS_STD_FRAC
    pts = centers[assign] + rng.normal(0.0, std, size=(num_nodes, 2))
    return np.clip(pts, -area, area)


def edge_biased(num_nodes, area, rng):
    """Radial density rising with distance from the origin (energy-hole stressor).

    Samples radius as ``area * u**(1/(EDGE_EXP+1))`` (mass pushed outward) and a
    uniform angle, then clips the square. Square clipping leaves a little extra
    mass in the corners, which is acceptable for a stress scenario.
    """
    u = rng.uniform(0.0, 1.0, size=num_nodes)
    r = area * u ** (1.0 / (EDGE_EXP + 1.0))
    theta = rng.uniform(0.0, 2.0 * math.pi, size=num_nodes)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return np.clip(pts, -area, area)


REGISTRY = {
    'poisson': poisson_disk,
    'uniform': uniform,
    'grid': grid,
    'gaussian': gaussian_clusters,
    'edge': edge_biased,
}


def generate_positions(name, num_nodes, area, rng):
    """Dispatch to the named generator. ``rng`` is owned by the caller."""
    try:
        generator = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown deployment '{name}'. Choices: {list(REGISTRY)}")
    return generator(num_nodes, area, rng)
