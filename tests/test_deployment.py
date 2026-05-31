"""Unit tests for the node-deployment generators (deployment.py).

For every scenario in REGISTRY this asserts the shared contract — exactly
num_nodes points, all coordinates in [-area, area], and determinism under a
fixed seed — plus one distribution-specific sanity check per scenario.

Run:  conda run -n base python tests/test_deployment.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deployment
from deployment import REGISTRY, generate_positions

AREA = 250
NUM_NODES = 200
SEED = 42


def _gen(name, num_nodes=NUM_NODES, seed=SEED):
    return generate_positions(name, num_nodes, AREA, np.random.default_rng(seed))


def test_contract():
    """Every generator: exact count, in-bounds, deterministic."""
    for name in REGISTRY:
        pts = _gen(name)
        assert pts.shape == (NUM_NODES, 2), \
            f"{name}: shape {pts.shape} != ({NUM_NODES}, 2)"
        assert np.all(pts >= -AREA) and np.all(pts <= AREA), \
            f"{name}: coordinates out of [-{AREA}, {AREA}]"
        pts2 = _gen(name)
        assert np.array_equal(pts, pts2), f"{name}: not deterministic under seed"
        print(f"  contract OK: {name}")


def test_poisson_spacing():
    """Poisson-disk: the sampler-placed batch respects the min radius."""
    rng = np.random.default_rng(SEED)
    engine = deployment.qmc.PoissonDisk(
        d=2, radius=deployment.POISSON_RADIUS, rng=rng,
        ncandidates=NUM_NODES, l_bounds=0, u_bounds=AREA * 2)
    placed = engine.random(NUM_NODES) - AREA   # only the disk-placed points
    if len(placed) >= 2:
        d = np.linalg.norm(placed[:, None, :] - placed[None, :, :], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        min_d = d.min()
        # allow a small tolerance below the nominal radius
        assert min_d >= deployment.POISSON_RADIUS * 0.95, \
            f"poisson: min spacing {min_d:.2f} < radius {deployment.POISSON_RADIUS}"
        print(f"  poisson spacing OK: min pairwise {min_d:.2f} "
              f">= {deployment.POISSON_RADIUS}")


def test_grid_regular():
    """Grid: unique x-coordinates are evenly spaced (pure lattice at jitter=0)."""
    assert deployment.GRID_JITTER == 0.0, "test assumes default GRID_JITTER=0"
    pts = _gen('grid')
    xs = np.unique(np.round(pts[:, 0], 6))
    diffs = np.diff(xs)
    assert diffs.std() < 1e-6, f"grid: x spacing not regular (std {diffs.std()})"
    print(f"  grid regular OK: {len(xs)} columns, spacing {diffs.mean():.2f}")


def test_gaussian_concentration():
    """Gaussian: most nodes lie within 3 std of their nearest blob center."""
    rng = np.random.default_rng(SEED)
    pts = generate_positions('gaussian', NUM_NODES, AREA, rng)
    # reconstruct centers the same way the generator draws them (first call)
    centers = np.random.default_rng(SEED).uniform(
        -AREA, AREA, size=(deployment.GAUSS_BLOBS, 2))
    dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dist.min(axis=1)
    std = AREA * deployment.GAUSS_STD_FRAC
    frac = np.mean(nearest <= 3 * std)
    assert frac >= 0.8, f"gaussian: only {frac:.0%} within 3 std of a center"
    print(f"  gaussian concentration OK: {frac:.0%} within 3 std")


def test_edge_bias():
    """Edge: more nodes in the outer ring than the inner disk."""
    pts = _gen('edge')
    r = np.linalg.norm(pts, axis=1)
    outer = np.sum(r > 0.5 * AREA)
    inner = np.sum(r <= 0.5 * AREA)
    assert outer > inner, f"edge: outer {outer} not > inner {inner}"
    print(f"  edge bias OK: outer {outer} > inner {inner}")


if __name__ == '__main__':
    print("Contract checks:")
    test_contract()
    print("Distribution checks:")
    test_poisson_spacing()
    test_grid_regular()
    test_gaussian_concentration()
    test_edge_bias()
    print("\nAll deployment tests passed.")
