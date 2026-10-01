# scenarios/freeze_coverage_scenarios.py
"""Snapshot the coverage-PSO deployment scenarios to scenarios/gen/*.csv.

Four configurations (two free-space, two with a central circular obstacle) are
each generated for every seed in benchmark.SEEDS. Each instance is an
independent projected-PSO placement, so instances are produced in parallel.
Re-running is idempotent (existing CSVs are skipped).

Run:  conda run -n base python -m scenarios.freeze_coverage_scenarios
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from benchmark import SEEDS
from .regions import Region
from .coverage_deploy import place_nodes
from .store import save_scenario

SCENARIO_DIR = os.path.join('scenarios', 'gen')
AREA = 250

# Each config: tag (file prefix), node count, coverage radius (= deployment comm
# range), circular obstacles [[cx, cy, r], ...], base-station position.
COVERAGE_CONFIGS = [
    {'tag': 'cov_free_n60_r60', 'num_nodes': 60, 'coverage_radius': 60,
     'circles': [], 'bs': [0.0, 0.0]},
    {'tag': 'cov_free_n40_r90', 'num_nodes': 40, 'coverage_radius': 90,
     'circles': [], 'bs': [0.0, 0.0]},
    {'tag': 'cov_obs_n60_r60', 'num_nodes': 60, 'coverage_radius': 60,
     'circles': [[0.0, 0.0, 120.0]], 'bs': [180.0, -180.0]},
    {'tag': 'cov_obs_n40_r90', 'num_nodes': 40, 'coverage_radius': 90,
     'circles': [[0.0, 0.0, 120.0]], 'bs': [180.0, -180.0]},
]


def _def_for(config, seed):
    """Build a region-definition dict for one (config, seed) instance."""
    return {
        'name': f"{config['tag']}_s{seed}",
        'area': config.get('area', AREA),
        'num_nodes': config['num_nodes'],
        'coverage_radius': config['coverage_radius'],
        'seed': seed,
        'bs': config['bs'],
        'obstacles': [],
        'circles': config['circles'],
        'paths': [],
    }


def freeze_one_coverage(config, seed, config_path='config/coverage.yaml',
                        out_dir=SCENARIO_DIR):
    """Generate one coverage scenario CSV. Returns (path, skipped)."""
    d = _def_for(config, seed)
    path = os.path.join(out_dir, f"{d['name']}.csv")
    if os.path.exists(path):
        return path, True

    region = Region.from_def(d)
    radius = float(d['coverage_radius'])
    positions = place_nodes(region, d['num_nodes'], radius, seed, config_path)
    rng = np.random.default_rng(seed)
    vpre = rng.uniform(2.7, 4.2, size=len(positions))

    os.makedirs(out_dir, exist_ok=True)
    save_scenario(path, positions, vpre, region.bs_pos,
                  {'source': config['tag'], 'num_nodes': len(positions),
                   'seed': seed, 'area': d['area'], 'coverage_radius': radius})
    return path, False


def main(workers=None):
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    jobs = [(c, s) for c in COVERAGE_CONFIGS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        futs = {ex.submit(freeze_one_coverage, c, s): (c['tag'], s)
                for c, s in jobs}
        for fut in as_completed(futs):
            tag, seed = futs[fut]
            path, skipped = fut.result()
            print(f"{'skip' if skipped else 'wrote'} {path}")


if __name__ == '__main__':
    main()
