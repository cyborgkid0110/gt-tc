# scenarios/freeze_scenarios.py
"""Snapshot the random deployment scenarios to scenarios/gen/*.csv.

Mirrors benchmark.py's sweep grid. Re-running is idempotent (existing files are
skipped). After freezing, benchmark.py consumes these CSVs instead of
regenerating.

Run:  conda run -n base python -m scenarios.freeze_scenarios
"""
import os

import numpy as np

from main import build_network
from .store import save_scenario

DEPLOYMENTS = ['poisson', 'uniform', 'grid', 'gaussian', 'edge']
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
NUM_NODES = 200
DEPLOYMENT_NODES = {'gaussian': 100}   # match benchmark.py overrides
SCENARIO_DIR = os.path.join('scenarios', 'gen')


def scenario_path(deployment, num_nodes, seed):
    return os.path.join(SCENARIO_DIR, f'{deployment}_n{num_nodes}_s{seed}.csv')


def freeze_one(deployment, num_nodes, seed):
    path = scenario_path(deployment, num_nodes, seed)
    if os.path.exists(path):
        return path, True
    net = build_network(deployment, num_nodes, seed)
    positions = np.array([[s.x, s.y] for s in net.sensors])
    vpre = np.array([s.Vpre for s in net.sensors])
    save_scenario(path, positions, vpre, (0.0, 0.0),
                  {'source': deployment, 'num_nodes': num_nodes, 'seed': seed,
                   'area': net.area, 'coverage_radius': ''})
    return path, False


def main():
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    for deployment in DEPLOYMENTS:
        n = DEPLOYMENT_NODES.get(deployment, NUM_NODES)
        for seed in SEEDS:
            path, skipped = freeze_one(deployment, n, seed)
            print(f"{'skip' if skipped else 'wrote'} {path}")


if __name__ == '__main__':
    main()
