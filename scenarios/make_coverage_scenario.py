"""Build a coverage scenario CSV from a region-definition YAML.

Run:  conda run -n base python -m scenarios.make_coverage_scenario --def scenarios/defs/example.yaml
Output: scenarios/gen/<name>.csv (consumed by build_network(scenario=...)).

`seed` may be a scalar (one CSV, <name>.csv) or a list (one CSV per seed,
<name>_s<seed>.csv). Multiple seeds are generated in parallel — each is an
independent projected-PSO placement.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from .regions import Region
from .coverage_deploy import place_nodes
from .store import save_scenario

SCENARIO_DIR = os.path.join('scenarios', 'gen')


def _generate_one(d, fname, seed, config_path='config/coverage.yaml',
                  out_dir=SCENARIO_DIR):
    """Generate one scenario CSV for a single seed. Returns (path, n_nodes).

    Rebuilds the Region from the def inside the worker so nothing shapely-shaped
    needs to be pickled across the process boundary.
    """
    region = Region.from_def(d)
    radius = float(d['coverage_radius'])
    positions = place_nodes(region, d['num_nodes'], radius, seed, config_path)
    rng = np.random.default_rng(seed)
    vpre = rng.uniform(2.7, 4.2, size=len(positions))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.csv")
    save_scenario(path, positions, vpre, region.bs_pos,
                  {'source': fname, 'num_nodes': len(positions),
                   'seed': seed, 'area': d['area'],
                   'coverage_radius': radius})
    return path, len(positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--def', dest='defn', required=True,
                    help='path to region-definition YAML')
    ap.add_argument('--name', dest='name', default=None,
                    help="output scenario name (overrides the YAML 'name'); "
                         "written to scenarios/gen/<name>.csv")
    ap.add_argument('--workers', type=int, default=None,
                    help='max parallel workers for multi-seed defs '
                         '(default: os.cpu_count())')
    args = ap.parse_args()

    with open(args.defn) as f:
        d = yaml.safe_load(f)

    name = args.name if args.name is not None else d['name']
    radius = float(d['coverage_radius'])   # comm range == coverage radius

    # Validate the region once up front (fail fast on a bad BS / obstacle) so we
    # don't spawn workers only to have each raise the same error.
    Region.from_def(d)

    # `seed` may be a single value or a list; a list generates one scenario per
    # seed, each written to <name>_s<seed>.csv. A scalar keeps <name>.csv.
    raw_seed = d['seed']
    multi = isinstance(raw_seed, (list, tuple))
    seeds = list(raw_seed) if multi else [raw_seed]
    jobs = [(seed, f"{name}_s{seed}" if multi else name) for seed in seeds]

    os.makedirs(SCENARIO_DIR, exist_ok=True)

    if len(jobs) == 1:
        seed, fname = jobs[0]
        path, n = _generate_one(d, fname, seed)
        print(f"wrote {path} ({n} nodes, "
              f"comm range = coverage radius = {radius:g} m)")
        return

    workers = min(args.workers or os.cpu_count(), len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_generate_one, d, fname, seed): (seed, fname)
                for seed, fname in jobs}
        for fut in as_completed(futs):
            path, n = fut.result()
            print(f"wrote {path} ({n} nodes, "
                  f"comm range = coverage radius = {radius:g} m)")


if __name__ == '__main__':
    main()
