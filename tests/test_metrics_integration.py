"""Integration test: metrics populated after a real run; family dispatch.

Run:  conda run -n base python tests/test_metrics_integration.py
"""
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_network
from algos import BaseAlgorithm
from algos.leach import LEACH


def test_metrics_populated_after_run():
    net = build_network('poisson', 50, 1)
    algo = LEACH(net, max_rounds=3, plot_period=10 ** 9)
    algo.run()

    assert len(algo.metrics.rounds) >= 1, 'no rounds recorded'
    assert algo.metrics.alive[0] <= 50
    s = algo.metrics.summary()
    # fnd uses pre-increment t, same convention as t_no_dead
    assert s['fnd'] == algo.t_no_dead, (s['fnd'], algo.t_no_dead)
    # LEACH is a clustering algorithm -> CH count series present
    assert 'ch_count' in algo.metrics.family, algo.metrics.family.keys()
    print('  metrics populated OK')


def test_family_dispatch_topology():
    net = build_network('poisson', 20, 1)

    class Dummy(BaseAlgorithm):
        family = 'topology'

        def _run_round(self):
            return False

    algo = Dummy(net, config_path='unused')
    extras = algo._collect_family_metrics()
    # avg_tx_power is now a universal-core metric (recorded in record_round),
    # so it is no longer a topology-family extra.
    assert set(extras) == {'avg_degree', 'lambda2'}, extras
    print('  topology dispatch OK')


def test_family_dispatch_default_empty():
    net = build_network('poisson', 20, 1)

    class Dummy(BaseAlgorithm):
        def _run_round(self):
            return False

    algo = Dummy(net, config_path='unused')
    assert algo._collect_family_metrics() == {}
    print('  default dispatch OK')


if __name__ == '__main__':
    test_metrics_populated_after_run()
    test_family_dispatch_topology()
    test_family_dispatch_default_empty()
    print('\nAll integration tests passed.')
