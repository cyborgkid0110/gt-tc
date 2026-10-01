"""Smoke test: plot_benchmark.generate_all produces PNG figures.

Run:  conda run -n base python tests/test_plot_smoke.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_benchmark


def _write_run(runs_dir, algo, deployment, seed):
    payload = {
        'algo': algo, 'deployment': deployment, 'seed': seed, 'num_nodes': 4,
        'summary': {
            'fnd': 5, 'hnd': 8, 'lnd': 10,
            'total_delivered': 30, 'total_generated': 40,
            'mean_pdr': 0.9, 'cumulative_pdr': 0.75,
            'energy_drained': 1.0, 'energy_per_packet': 0.03,
            'mean_energy_std': 0.1,
            'mean_avg_hop': 2.0, 'mean_avg_tx_power': 1e-4,
        },
        'time_series': {
            'rounds': [0, 1, 2], 'alive': [4, 4, 3],
            'total_energy': [40.0, 38.0, 36.0], 'energy_std': [0.0, 0.0, 0.1],
            'delivered': [4, 4, 3], 'generated': [4, 4, 3],
            'avg_hop': [2.0, 2.5, 2.0], 'avg_tx_power': [1e-4, 1e-4, 1e-4],
            'family': {'ch_count': [1, 1, 1], 'cluster_sizes': [[2], [2], [2]]},
        },
    }
    with open(os.path.join(runs_dir, f'{algo}_{deployment}_{seed}.json'),
              'w') as f:
        json.dump(payload, f)


def test_generate_all_creates_pngs():
    d = tempfile.mkdtemp()
    runs_dir = os.path.join(d, 'runs')
    os.makedirs(runs_dir)
    _write_run(runs_dir, 'LEACH', 'poisson', 1)
    _write_run(runs_dir, 'LEACH', 'poisson', 2)
    _write_run(runs_dir, 'GT2', 'poisson', 1)

    plot_benchmark.generate_all(d)

    fig_dir = os.path.join(d, 'figures')
    assert os.path.isdir(fig_dir), fig_dir
    pngs = [fn for fn in os.listdir(fig_dir) if fn.endswith('.png')]
    assert pngs, 'no PNG figures produced'
    assert 'survival_poisson.png' in pngs, pngs
    assert 'fnd.png' in pngs, pngs
    print('  figures produced:', sorted(pngs))

    deployments = ['poisson']
    for dep in deployments:
        assert os.path.exists(os.path.join(fig_dir, f'hop_{dep}.png')), \
            f'missing hop_{dep}.png'
        assert os.path.exists(os.path.join(fig_dir, f'tx_power_{dep}.png')), \
            f'missing tx_power_{dep}.png'
        assert os.path.exists(os.path.join(fig_dir, f'ch_count_{dep}.png')), \
            f'missing ch_count_{dep}.png'
    with open(os.path.join(d, 'summary_by_scenario.csv')) as f:
        header = f.readline()
    assert 'energy_per_packet_median' in header, \
        f'energy_per_packet_median not in CSV header: {header}'
    for col in ('fnd_ci95', 'fnd_q1', 'fnd_q3', 'fnd_median'):
        assert col in header, f'{col} not in CSV header: {header}'
    assert 'boxplot_fnd_poisson.png' in pngs, pngs
    print('  new figures + median/CI/IQR columns + box plots OK')


if __name__ == '__main__':
    test_generate_all_creates_pngs()
    print('\nPlot smoke test passed.')
