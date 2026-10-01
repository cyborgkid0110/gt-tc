"""Render benchmark figures from persisted run JSON (never re-runs the sweep).

Reads results/runs/*.json and writes results/figures/*.png.

Run:  conda run -n base python plot_benchmark.py
"""
import csv
import glob
import json
import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

from plot_style import (DPI, PANEL_SIZE, WIDE_SIZE, apply_paper_style,
                        colour_map, scenario_panel_size, thin_xticks)

apply_paper_style()

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
# Short y labels for the box-plot panels: the full SCALAR_METRICS phrases do
# not fit a 3.1 in tall panel at paper font sizes.
BOX_YLABELS = {
    'fnd': 'FND (round)',
    'hnd': 'HND (round)',
    'lnd': 'LND (round)',
    'total_delivered': 'Packets to BS',
    'cumulative_pdr': 'PDR',
    'energy_per_packet': 'J / packet',
    'mean_avg_hop': r'$H_\mathrm{avg}$',
    'mean_avg_tx_power': 'Avg. tx power',
}


def load_runs(results_dir=RESULTS_DIR):
    runs = []
    for fn in glob.glob(os.path.join(results_dir, 'runs', '*.json')):
        with open(fn) as f:
            runs.append(json.load(f))
    return runs


def _aligned_mean_std(series_list):
    """Mean/std across runs of differing length; pad short runs with last value.

    Series may contain None (e.g. avg_hop on a no-delivery round); numpy stores
    those as NaN. A round where every run is NaN (no seed delivered) yields NaN —
    plotted as a gap. nanmean/nanstd raise an expected "Mean of empty slice"
    RuntimeWarning on such all-NaN columns; suppress it.
    """
    length = max(len(s) for s in series_list)
    arr = np.full((len(series_list), length), np.nan)
    for i, s in enumerate(series_list):
        if not s:
            arr[i, :] = 0.0
            continue
        arr[i, :len(s)] = s
        arr[i, len(s):] = s[-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _curve(runs, deployment, ts_key, ylabel, fname, fig_dir, *, family_key=None,
           legend=True, title=True):
    """Per-deployment mean±std over-time curve, one line per algorithm.

    Reads a top-level time-series key (`ts_key`) by default, or a family-extra
    series when `family_key` is given (skipping runs that lack the family key —
    e.g. topology algorithms have no `ch_count`).

    `legend=False` / `title=False` drop the per-panel legend and title: the
    paper tiles six of these panels into one figure, where a repeated 11-entry
    legend wastes most of the page. Use `legend_strip()` for a shared legend
    and the LaTeX subfigure captions for the panel names.
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
    # The tiled paper panels (legend=False) follow the PANEL_COLS layout; the
    # standalone diagnostic curves keep the wider two-per-row size.
    plt.figure(figsize=scenario_panel_size() if not legend else PANEL_SIZE)
    colours = colour_map(sorted(by_algo))
    for algo, series_list in sorted(by_algo.items()):
        mean, std = _aligned_mean_std(series_list)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=algo, color=colours[algo])
        plt.fill_between(x, mean - std, mean + std, alpha=0.15,
                         color=colours[algo])
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    thin_xticks()
    if title:
        plt.title(deployment)
    if legend:
        # Below the axes: 11 algorithms occlude the curves if placed inside.
        # No tight_layout — it ignores the out-of-axes legend; plot_style's
        # savefig(bbox='tight') expands the canvas instead.
        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.24),
                   frameon=False, fontsize=9, handlelength=1.2,
                   columnspacing=0.8, handletextpad=0.4)
    else:
        plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=DPI)
    plt.close()


def legend_strip(runs, fname, fig_dir):
    """Standalone horizontal legend shared by the tiled per-scenario panels.

    Colours match `_curve` (both key off the sorted algorithm names).
    """
    algos = sorted({r['algo'] for r in runs})
    if not algos:
        return
    colours = colour_map(algos)
    fig = plt.figure(figsize=(WIDE_SIZE[0], 0.5))
    handles = [Line2D([], [], color=colours[a], lw=2.2, label=a) for a in algos]
    fig.legend(handles=handles, loc='center', ncol=6, frameon=False,
               fontsize=9, handlelength=1.4, columnspacing=1.0,
               handletextpad=0.4)
    fig.savefig(os.path.join(fig_dir, fname), dpi=DPI)
    plt.close(fig)


def _boxplot(runs, deployment, key, ylabel, fname, fig_dir):
    """Per-deployment distribution of a scalar metric over seeds.

    One box per algorithm (median, IQR box, 1.5·IQR whiskers, fliers) with the
    mean as a marker, so run-to-run uncertainty is visible rather than folded
    into a single ±std bar. Runs whose metric is None (e.g. LND when the
    network never fully dies) are dropped. Colours key off the algorithms
    present in the deployment, matching `_curve`. Like the tiled survival /
    energy panels there is no title (the LaTeX subfigure caption names the
    scenario) and no legend (algorithm names are the x tick labels).
    """
    dep_runs = [r for r in runs if r['deployment'] == deployment]
    colours = colour_map(sorted({r['algo'] for r in dep_runs}))
    by_algo = defaultdict(list)
    for r in dep_runs:
        v = r['summary'].get(key)
        if v is not None:
            by_algo[r['algo']].append(v)
    if not by_algo:
        return
    algos = sorted(by_algo)

    fig, ax = plt.subplots(figsize=scenario_panel_size())
    bp = ax.boxplot([by_algo[a] for a in algos], patch_artist=True,
                    showmeans=True, widths=0.6,
                    medianprops=dict(color='black', lw=1.0),
                    meanprops=dict(marker='^', markersize=3.5,
                                   markerfacecolor='white',
                                   markeredgecolor='black', markeredgewidth=0.6),
                    flierprops=dict(marker='o', markersize=2.0, alpha=0.5),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, algo in zip(bp['boxes'], algos):
        patch.set_facecolor(colours[algo])
        patch.set_alpha(0.75)
        patch.set_linewidth(0.8)
    # Set tick labels directly: boxplot's `labels` kwarg was renamed to
    # `tick_labels` in matplotlib 3.9.
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos, rotation=60, ha='right',
                       rotation_mode='anchor', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, fname), dpi=DPI)
    plt.close(fig)


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
    plt.figure(figsize=WIDE_SIZE)
    plt.bar(range(len(algos)), means, yerr=stds, capsize=3)
    plt.xticks(range(len(algos)), algos, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=DPI)
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
    plt.figure(figsize=(max(WIDE_SIZE[0], len(algos) * 0.8), WIDE_SIZE[1]))
    for i, dep in enumerate(deployments):
        means = [np.mean(cell[(a, dep)]) if cell[(a, dep)] else 0.0
                 for a in algos]
        stds = [np.std(cell[(a, dep)]) if cell[(a, dep)] else 0.0
                for a in algos]
        plt.bar(x + i * width, means, width, yerr=stds, capsize=2, label=dep)
    plt.xticks(x + 0.4 - width / 2, algos, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} — by scenario')
    plt.legend(title='deployment')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=DPI)
    plt.close()


def _ci95_halfwidth(vals):
    """Half-width of the t-distribution 95 % CI of the mean ('' if n < 2)."""
    n = len(vals)
    if n < 2:
        return ''
    sem = np.std(vals, ddof=1) / np.sqrt(n)
    return float(student_t.ppf(0.975, n - 1) * sem)


def write_scenario_summary(runs, results_dir):
    """Aggregate per (deployment, algo) statistics of each scalar metric.

    Writes results/summary_by_scenario.csv — one row per (deployment, algo),
    answering "what are the metrics of each scenario" directly. Per metric:
    mean, std, `ci95` (half-width of the 95 % CI of the mean), and the box-plot
    statistics median / q1 / q3 (the `boxplot_*` figures show the same data).
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
        fields += [f'{key}_{s}'
                   for s in ('mean', 'std', 'ci95', 'median', 'q1', 'q3')]

    path = os.path.join(results_dir, 'summary_by_scenario.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (dep, algo) in sorted(cell):
            c = cell[(dep, algo)]
            row = {'deployment': dep, 'algo': algo, 'n_seeds': len(c['seeds'])}
            for key, _ in SCALAR_METRICS:
                vals = c[key]
                if not vals:
                    for s in ('mean', 'std', 'ci95', 'median', 'q1', 'q3'):
                        row[f'{key}_{s}'] = ''
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                row[f'{key}_mean'] = float(np.mean(vals))
                row[f'{key}_std'] = float(np.std(vals))
                row[f'{key}_ci95'] = _ci95_halfwidth(vals)
                row[f'{key}_median'] = float(med)
                row[f'{key}_q1'] = float(q1)
                row[f'{key}_q3'] = float(q3)
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
        # Short axis labels: the full metric names do not fit a 3.1 in tall
        # panel at paper font sizes (the paper's caption names the metric).
        # survival_/energy_ are tiled 6-up in the paper: no per-panel legend or
        # title (LaTeX subfigure captions name the scenario, and
        # `algo_legend.png` carries the shared legend).
        _curve(runs, deployment, 'alive', 'Alive nodes',
               f'survival_{deployment}.png', fig_dir,
               legend=False, title=False)
        _curve(runs, deployment, 'total_energy', r'$E_\mathrm{res}$ (J)',
               f'energy_{deployment}.png', fig_dir,
               legend=False, title=False)
        _curve(runs, deployment, 'avg_hop', r'$H_\mathrm{avg}$',
               f'hop_{deployment}.png', fig_dir)
        _curve(runs, deployment, 'avg_tx_power', 'Avg. tx power',
               f'tx_power_{deployment}.png', fig_dir)
        _curve(runs, deployment, None, 'CH count',
               f'ch_count_{deployment}.png', fig_dir, family_key='ch_count')
        # Seed-to-seed uncertainty of each scalar metric, tiled like the
        # survival_/energy_ panels.
        for key, label in SCALAR_METRICS:
            _boxplot(runs, deployment, key, BOX_YLABELS.get(key, label),
                     f'boxplot_{key}_{deployment}.png', fig_dir)

    legend_strip(runs, 'algo_legend.png', fig_dir)

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
