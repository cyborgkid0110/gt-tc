# tests/test_coverage_deploy.py
"""Projected-PSO coverage placement: in-region, and connected AT the single
deployment radius (communication range == coverage radius)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.regions import Region
from scenarios.coverage_deploy import place_nodes, is_connected_to_bs

# A small, fast PSO config so the test does not depend on the tuned
# config/coverage.yaml (and runs quickly regardless of its iteration count).
_FAST_CFG = """
pso_particles: 20
pso_iters: 40
pso_w: 0.7
pso_c1: 1.5
pso_c2: 1.5
penalty_lambda: 2.0
target_spacing: 10
"""


def _cfg_path(tmpdir):
    p = os.path.join(tmpdir, 'coverage.yaml')
    with open(p, 'w') as f:
        f.write(_FAST_CFG)
    return p


def _open_region():
    return Region.from_def({
        'area': 100, 'num_nodes': 12, 'coverage_radius': 40, 'seed': 1,
        'bs': [0.0, 0.0], 'obstacles': [], 'paths': [],
    })


def test_connected_at_coverage_radius():
    """The deployment must be connected at the SAME radius used to place it —
    i.e. communication range == coverage radius (the property this guarantees)."""
    region = _open_region()
    radius = 40.0
    with tempfile.TemporaryDirectory() as d:
        positions = place_nodes(region, num_nodes=12, radius=radius,
                                seed=1, config_path=_cfg_path(d))
    assert positions.shape == (12, 2), "wrong node count"
    from shapely.geometry import Point
    for p in positions:
        # projection may land on the boundary, so allow a hair of tolerance
        assert region.deployable.distance(Point(p)) < 1e-6, f"node {p} outside region"
    assert is_connected_to_bs(positions, region.bs_pos, radius), \
        "deployment is not connected at the coverage radius"
    print("  connected at coverage radius OK")


def test_determinism():
    region = _open_region()
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg_path(d)
        a = place_nodes(region, 12, 40.0, seed=5, config_path=cfg)
        b = place_nodes(region, 12, 40.0, seed=5, config_path=cfg)
    assert np.allclose(a, b), "same seed must give same layout"
    print("  determinism OK")


if __name__ == '__main__':
    test_connected_at_coverage_radius()
    test_determinism()
    print("\nAll coverage-deploy tests passed.")
