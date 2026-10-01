# tests/test_freeze_coverage.py
"""Single-instance coverage generation: in-region, obstacle-free, BS-connected.

Run:  conda run -n base python tests/test_freeze_coverage.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import load_scenario
from scenarios.coverage_deploy import is_connected_to_bs
from scenarios.freeze_coverage_scenarios import freeze_one_coverage

# Fast PSO config so the test does not depend on the tuned config/coverage.yaml.
_FAST_CFG = """
pso_particles: 20
pso_iters: 40
pso_w: 0.7
pso_c1: 1.5
pso_c2: 1.5
penalty_lambda: 2.0
target_spacing: 10
"""


def test_freeze_one_coverage_with_obstacle():
    config = {'tag': 'cov_obs_test', 'num_nodes': 12, 'coverage_radius': 40,
              'circles': [[0.0, 0.0, 30.0]], 'bs': [80.0, -80.0], 'area': 100}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, 'coverage.yaml')
        with open(cfg, 'w') as f:
            f.write(_FAST_CFG)
        path, skipped = freeze_one_coverage(config, seed=1,
                                            config_path=cfg, out_dir=tmp)
        assert not skipped, "first generation should not be skipped"
        assert os.path.exists(path), path

        positions, vpre, bs_pos, meta = load_scenario(path)
        assert positions.shape == (12, 2), positions.shape
        # no node lands inside the central circular obstacle
        dist = np.hypot(positions[:, 0], positions[:, 1])
        assert (dist > 30.0 - 1e-6).all(), "a node is inside the obstacle"
        # BS persisted correctly
        assert abs(bs_pos[0] - 80.0) < 1e-9 and abs(bs_pos[1] + 80.0) < 1e-9
        # connected at the coverage radius
        assert is_connected_to_bs(positions, bs_pos, 40.0), "not BS-connected"
        # meta carries the provenance
        assert meta['source'] == 'cov_obs_test'
        assert int(meta['num_nodes']) == 12

        # second call is idempotent (file exists -> skipped)
        _path2, skipped2 = freeze_one_coverage(config, seed=1,
                                               config_path=cfg, out_dir=tmp)
        assert skipped2, "second generation should be skipped"
    print("  freeze_one_coverage (obstacle) OK")


if __name__ == '__main__':
    test_freeze_one_coverage_with_obstacle()
    print("\nAll freeze-coverage tests passed.")
