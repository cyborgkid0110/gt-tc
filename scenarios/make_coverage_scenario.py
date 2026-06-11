"""Build a coverage scenario CSV from a region-definition YAML.

Run:  conda run -n base python -m scenarios.make_coverage_scenario --def scenarios/defs/example.yaml
Output: scenarios/gen/<name>.csv (consumed by build_network(scenario=...)).
"""
import argparse
import os

import numpy as np
import yaml

from .regions import Region
from .coverage_deploy import place_nodes
from .store import save_scenario

SCENARIO_DIR = os.path.join('scenarios', 'gen')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--def', dest='defn', required=True,
                    help='path to region-definition YAML')
    args = ap.parse_args()

    with open(args.defn) as f:
        d = yaml.safe_load(f)

    region = Region.from_def(d)
    radius = float(d['coverage_radius'])   # comm range == coverage radius
    rng = np.random.default_rng(d['seed'])

    positions = place_nodes(region, d['num_nodes'], radius,
                            d['seed'], 'config/coverage.yaml')
    vpre = rng.uniform(2.7, 4.2, size=len(positions))

    os.makedirs(SCENARIO_DIR, exist_ok=True)
    out = os.path.join(SCENARIO_DIR, f"{d['name']}.csv")
    save_scenario(out, positions, vpre, region.bs_pos,
                  {'source': d['name'], 'num_nodes': len(positions),
                   'seed': d['seed'], 'area': d['area'],
                   'coverage_radius': d['coverage_radius']})
    print(f"wrote {out} ({len(positions)} nodes, "
          f"comm range = coverage radius = {radius:g} m)")


if __name__ == '__main__':
    main()
