"""Smoke test: clustering algos run a few rounds and drain energy.
Run:  conda run -n base python tests/test_cluster_smoke.py
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_network, make_algo
from benchmark import _disable_plotting

CLUSTERING = ['GT2', 'LEACH', 'GTFR', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA']


def test_clustering_algos_run_and_drain():
    for algo in CLUSTERING:
        # Each algo needs its own fresh network (runs mutate energy).
        net = build_network(deployment='poisson', num_nodes=100, seed=7)
        e0 = sum(s.e_res for s in net.sensors)
        a = make_algo(algo, net, dict(max_rounds=3, plot_period=10 ** 9))
        _disable_plotting()
        a.run()
        e1 = sum(s.e_res for s in net.sensors)
        assert e1 < e0, f'{algo}: energy did not decrease'
        ts = a.metrics.time_series()
        assert len(ts['avg_hop']) >= 1, f'{algo}: no rounds recorded'
        print(f'{algo}: ok (drained {e0 - e1:.4f} J)')


if __name__ == '__main__':
    test_clustering_algos_run_and_drain()
    print('Task 4 smoke passed')
