"""Render benchmark figures from persisted run JSON (never re-runs the sweep).

Reads results/runs/*.json and writes results/figures/*.png.

Run:  conda run -n base python plot_benchmark.py
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'results'

# Scalar metrics summarised per scenario (CSV columns + bar charts).
SCALAR_METRICS = [
    ('fnd', 'First Node Death (round)'),
    ('hnd', 'Half Node Death (round)'),
    ('lnd', 'Last Node Death (round)'),
    ('total_delivered', 'Total packets to BS'),
    ('cumulative_pdr', 'Cumulative PDR'),
    ('energy_per_packet', 'Energy per delivered packet (J)'),
    ('mean_avg_hop', 'Average hop count to BS'),
    ('mean_avg_tx_power', 'Average transmit power per node'),
]


def load_runs(results_dir=RESULTS_DIR):
    runs = []
    for fn in glob.glob(os.path.join(results_dir, 'runs', '*.json')):
        with open(fn) as f:
            runs.append(json.load(f))
    return runs


def _aligned_mean_std(series_list):
    """Mean/std across runs of differing length; pad short runs with last value."""
    length = max(len(s) for s in series_list)
    arr = np.full((len(series_list), length), np.nan)
    for i, s in enumerate(series_list):
        if not s:
            arr[i, :] = 0.0
            continue
        arr[i, :len(s)] = s
        arr[i, len(s):] = s[-1]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _curve(runs, deployment, ts_key, ylabel, fname, fig_dir, *, family_key=None):
    """Per-deployment mean±std over-time curve, one line per algorithm.

    Reads a top-level time-series key (`ts_key`) by default, or a family-extra
    series when `family_key` is given (skipping runs that lack the family key —
    e.g. topology algorithms have no `ch_count`).
    """
    by_algo = defaultdict(list)
    for r in runs:
        if r['deployment'] != deployment:
            continue
        if family_key is not None:
            series = r['time_series'].get('family', {}).get(family_key)
        else:
            series = r['time_series'].get(ts_key)
        if series is not None:
            by_algo[r['algo']].append(series)
    if not by_algo:
        return
    plt.figure(figsize=(8, 5))
    for algo, series_list in sorted(by_algo.items()):
        mean, std = _aligned_mean_std(series_list)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} — {deployment}')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=120)
    plt.close()


def _bar_metric(runs, key, ylabel, fname, fig_dir):
    by_algo = defaultdict(list)
    for r in runs:
        v = r['summary'].get(key)
        if v is None:
            continue
        by_algo[r['algo']].append(v)
    if not by_algo:
        return
    algos = sorted(by_algo)
    means = [np.mean(by_algo[a]) for a in algos]
    stds = [np.std(by_algo[a]) for a in algos]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(algos)), means, yerr=stds, capsize=3)
    plt.xticks(range(len(algos)), algos, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=120)
    plt.close()


def _bar_metric_by_scenario(runs, key, ylabel, fname, fig_dir):
    """Grouped bar chart: one bar group per algorithm, one bar per deployment.

    Unlike `_bar_metric` (which averages a metric over *all* deployments), this
    keeps the scenarios separate so per-scenario differences are visible.
    """
    # (algo, deployment) -> list of run values
    cell = defaultdict(list)
    deployments = set()
    for r in runs:
        v = r['summary'].get(key)
        if v is None:
            continue
        cell[(r['algo'], r['deployment'])].append(v)
        deployments.add(r['deployment'])
    if not cell:
        return
    algos = sorted({a for a, _ in cell})
    deployments = sorted(deployments)

    x = np.arange(len(algos))
    width = 0.8 / len(deployments)
    plt.figure(figsize=(max(10, len(algos) * 1.1), 5.5))
    for i, dep in enumerate(deployments):
        means = [np.mean(cell[(a, dep)]) if cell[(a, dep)] else 0.0
                 for a in algos]
        stds = [np.std(cell[(a, dep)]) if cell[(a, dep)] else 0.0
                for a in algos]
        plt.bar(x + i * width, means, width, yerr=stds, capsize=2, label=dep)
    plt.xticks(x + 0.4 - width / 2, algos, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} — by scenario')
    plt.legend(title='deployment', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=120)
    plt.close()


def write_scenario_summary(runs, results_dir):
    """Aggregate per (deployment, algo): mean & std of each scalar metric.

    Writes results/summary_by_scenario.csv — one row per (deployment, algo),
    answering "what are the metrics of each scenario" directly (the bar-chart
    figures show the same data graphically).
    """
    cell = defaultdict(lambda: defaultdict(list))
    for r in runs:
        c = cell[(r['deployment'], r['algo'])]
        c['seeds'].append(r['seed'])
        for key, _ in SCALAR_METRICS:
            v = r['summary'].get(key)
            if v is not None:
                c[key].append(v)

    fields = ['deployment', 'algo', 'n_seeds']
    for key, _ in SCALAR_METRICS:
        fields += [f'{key}_mean', f'{key}_std']
    fields += ['energy_per_packet_median']

    path = os.path.join(results_dir, 'summary_by_scenario.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (dep, algo) in sorted(cell):
            c = cell[(dep, algo)]
            row = {'deployment': dep, 'algo': algo, 'n_seeds': len(c['seeds'])}
            for key, _ in SCALAR_METRICS:
                vals = c[key]
                row[f'{key}_mean'] = float(np.mean(vals)) if vals else ''
                row[f'{key}_std'] = float(np.std(vals)) if vals else ''
            epp = c['energy_per_packet']
            row['energy_per_packet_median'] = (float(np.median(epp))
                                               if epp else '')
            w.writerow(row)
    return path


def generate_all(results_dir=RESULTS_DIR):
    runs = load_runs(results_dir)
    if not runs:
        print('No runs found; nothing to plot.')
        return
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for deployment in sorted({r['deployment'] for r in runs}):
        _curve(runs, deployment, 'alive', 'Alive nodes',
               f'survival_{deployment}.png', fig_dir)
        _curve(runs, deployment, 'total_energy', 'Total residual energy (J)',
               f'energy_{deployment}.png', fig_dir)
        _curve(runs, deployment, 'avg_hop', 'Average hop count to BS',
               f'hop_{deployment}.png', fig_dir)
        _curve(runs, deployment, 'avg_tx_power',
               'Average transmit power per node',
               f'tx_power_{deployment}.png', fig_dir)
        _curve(runs, deployment, None, 'Cluster-head count',
               f'ch_count_{deployment}.png', fig_dir, family_key='ch_count')

    _bar_metric(runs, 'fnd', 'First Node Death (round)', 'fnd.png', fig_dir)
    _bar_metric(runs, 'hnd', 'Half Node Death (round)', 'hnd.png', fig_dir)
    _bar_metric(runs, 'lnd', 'Last Node Death (round)', 'lnd.png', fig_dir)
    _bar_metric(runs, 'total_delivered', 'Total packets to BS',
                'delivered.png', fig_dir)
    _bar_metric(runs, 'cumulative_pdr', 'Cumulative PDR', 'pdr.png', fig_dir)
    _bar_metric(runs, 'energy_per_packet', 'Energy per delivered packet (J)',
                'energy_per_packet.png', fig_dir)

    # Per-scenario breakdown (keeps deployments separate, not averaged).
    for key, label in SCALAR_METRICS:
        _bar_metric_by_scenario(runs, key, label, f'{key}_by_scenario.png',
                                fig_dir)
    csv_path = write_scenario_summary(runs, results_dir)

    print(f'Figures written to {fig_dir}')
    print(f'Per-scenario summary written to {csv_path}')


if __name__ == '__main__':
    generate_all()
