"""Verify BS-connectivity of the benchmark scenarios at the *initial* transmit
power (the power nodes are loaded with, before any algorithm adjusts it).

The load gate (`NetworkModel.check_potential_connectivity`) only guarantees a
layout is connectable at *max* power. This script answers a stricter question:
is every node already able to reach the base station at the *initial* power the
simulation starts from? It iterates the same `SCENARIOS x SEEDS` grid as
`benchmark.py`, so it reflects exactly what the sweep runs.

For each scenario it also reports the minimum comm range (and the power that
yields it) needed to BS-connect every node — i.e. the initial power you would
have to set for connectivity to hold from round 0.

Run:  conda run -n base python check_initial_connectivity.py
      conda run -n base python check_initial_connectivity.py --power-frac 0.5
"""
import argparse
import os

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from benchmark import SCENARIOS, SEEDS, scenario_csv
from main import build_network, P_MAX, P_MIN
from model import Sensor, NetworkModel


def _bs_connected(positions, bs_pos, rc):
    """True iff every node reaches the BS at range rc (BS-rooted, multi-hop)."""
    pts = np.vstack([np.asarray(bs_pos, dtype=float), positions])
    seen = np.zeros(len(pts), dtype=bool)
    seen[0] = True
    frontier = [0]
    while frontier:
        i = frontier.pop()
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        nbrs = np.nonzero((d <= rc) & ~seen)[0]
        seen[nbrs] = True
        frontier.extend(nbrs.tolist())
    return bool(seen[1:].all())


def _min_bs_range(positions, bs_pos):
    """Smallest range that BS-connects every node = max edge of MST(nodes+BS)."""
    pts = np.vstack([np.asarray(bs_pos, dtype=float), positions])
    mst = minimum_spanning_tree(cdist(pts, pts)).toarray()
    return float(mst[mst > 0].max())


def _range_for_power(power):
    """Physics comm range at a given transmit power."""
    s = Sensor(id=0, x=0.0, y=0.0, e0=0.5, power=power, Vpre=3.0)
    net = NetworkModel([s], 250, p_max=P_MAX, p_min=P_MIN)
    return net.calc_comm_range(power)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--power-frac', type=float, default=None,
                    help='override: test a uniform initial power as a fraction '
                         'of P_MAX (default: use each scenario\'s actual loaded '
                         'initial power, which build_network sets from '
                         'coverage_radius)')
    args = ap.parse_args()

    if args.power_frac is None:
        print(f'Initial power = each scenario\'s actual loaded power '
              f'(comm range = coverage_radius where defined, else P_MAX/4)   '
              f'(P_MAX range = {_range_for_power(P_MAX):.1f} m)\n')
        override_range = None
    else:
        override_range = _range_for_power(args.power_frac * P_MAX)
        print(f'Initial power = {args.power_frac:g} * P_MAX (uniform override)  '
              f'->  comm range = {override_range:.1f} m   '
              f'(P_MAX range = {_range_for_power(P_MAX):.1f} m)\n')

    hdr = f'{"scenario":<20} {"init range":>11} {"connected@init":>16} {"worst min-range":>16} {"missing":>9}'
    print(hdr)
    print('-' * len(hdr))

    overall_worst = 0.0
    for tag in SCENARIOS:
        ok = 0
        worst = 0.0
        missing = 0
        rc_init = None
        for seed in SEEDS:
            csv = scenario_csv(tag, seed)
            if not os.path.exists(csv):
                missing += 1
                continue
            net = build_network(scenario=csv)
            rc_init = net.sensors[0].rc if override_range is None else override_range
            positions = np.array([[s.x, s.y] for s in net.sensors], dtype=float)
            bs = (net.bs_x, net.bs_y)
            if _bs_connected(positions, bs, rc_init):
                ok += 1
            worst = max(worst, _min_bs_range(positions, bs))
        overall_worst = max(overall_worst, worst)
        n = len(SEEDS) - missing
        rc_str = f'{rc_init:.1f} m' if rc_init is not None else '-'
        print(f'{tag:<20} {rc_str:>11} {f"{ok}/{n}":>16} {f"{worst:.1f} m":>16} '
              f'{missing if missing else "":>9}')

    print('-' * len(hdr))
    print(f'\nTo BS-connect EVERY scenario at initial power, the initial comm '
          f'range must be >= {overall_worst:.1f} m.')
    if overall_worst > _range_for_power(P_MAX):
        print(f'  (NOTE: that exceeds the P_MAX range {_range_for_power(P_MAX):.1f} m '
              f'— not reachable without raising P_MAX.)')


if __name__ == '__main__':
    main()
