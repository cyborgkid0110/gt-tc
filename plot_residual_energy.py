"""Residual-energy-over-time figures, one per scenario, truncated at each
algorithm's first node death (FND).

Same style as the benchmark `energy_{deployment}.png` curves (mean +/- std,
one line per algorithm) but every algorithm's line is drawn only up to its
mean FND round -- i.e. while the network is still fully alive -- and the panel
title uses the paper's scenario name (e.g. "Scenario A: Free space, N = 200").

Reads results/runs/*.json. Writes results/figures/energy_before_fnd_*.png and
mirrors them into the paper tree.

Run:  conda run -n base python plot_residual_energy.py
"""
import glob
import json
import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_style import (DPI, apply_paper_style, colour_map,
                        scenario_panel_size, thin_xticks)

apply_paper_style()

RESULTS_DIR = 'results'
# Paper tree to mirror figures into; override with PAPER_DIR for the
# Elsevier-template copy of the paper (els-cas-templates).
PAPER_DIR = os.environ.get(
    'PAPER_DIR',
    os.path.expanduser('~/Documents/workspace/github/GT2_paper'))
PAPER_FIG_DIR = os.path.join(PAPER_DIR, 'figures/result/energy_efficiency')

# deployment key -> paper scenario title
SCENARIO_TITLE = {
    'uniform_n200':     r'Scenario A: Free space, $N = 200$',
    'gaussian_n100':    r'Scenario B: Target region, $N = 100$',
    'cov_free_n60_r60': r'Scenario C: Free space, $N = 60$',
    'cov_free_n40_r90': r'Scenario D: Free space, $N = 40$',
    'cov_obs_n60_r60':  r'Scenario E: Obstacle, $N = 60$',
    'cov_obs_n40_r90':  r'Scenario F: Obstacle, $N = 40$',
}
# fixed order so the legend/colours are consistent across panels
ALGO_ORDER = ['GT2', 'LEACH', 'GTFR', 'DIA', 'MIA', 'TCLE', 'EFTCG-1',
              'EFTCG-2', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA']


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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def residual_energy_before_fnd(runs, deployment, fig_dir):
    """One panel: mean +/- std total residual energy vs round, each algorithm's
    line truncated at its mean FND."""
    energy = defaultdict(list)
    fnds = defaultdict(list)
    for r in runs:
        if r['deployment'] != deployment:
            continue
        series = r['time_series'].get('total_energy')
        fnd = r['summary'].get('fnd')
        if series is not None and fnd is not None:
            energy[r['algo']].append(series)
            fnds[r['algo']].append(fnd)
    if not energy:
        return None

    # ALGO_ORDER controls the legend (GT2 first); colours come from the sorted
    # names so an algorithm keeps the same colour as in plot_benchmark.py.
    algos = [a for a in ALGO_ORDER if a in energy] + \
            [a for a in sorted(energy) if a not in ALGO_ORDER]
    colours = colour_map(sorted(energy))

    plt.figure(figsize=scenario_panel_size())
    for algo in algos:
        mean, std = _aligned_mean_std(energy[algo])
        cut = int(round(np.mean(fnds[algo])))          # mean FND round
        cut = max(1, min(cut, len(mean)))              # keep >= 1 sample
        x = np.arange(cut)
        m, s = mean[:cut], std[:cut]
        lw = 2.0 if algo == 'GT2' else 1.1
        z = 5 if algo == 'GT2' else 2
        plt.plot(x, m, label=algo, color=colours[algo], linewidth=lw, zorder=z)
        plt.fill_between(x, m - s, m + s, color=colours[algo], alpha=0.12,
                         zorder=1)
        # mark the FND cut-off point
        plt.scatter([x[-1]], [m[-1]], color=colours[algo], s=18, zorder=z + 1)

    plt.xlabel('Round')
    # Short label: the full phrase does not fit a 3.1 in tall panel at this
    # font size. Matches the paper's E_res notation.
    plt.ylabel(r'$E_\mathrm{res}$ (J)')
    thin_xticks()
    # No per-panel title or legend: the paper tiles six panels into one figure,
    # where the LaTeX subfigure caption names the scenario and
    # `algo_legend.png` (plot_benchmark.legend_strip) carries a shared legend.
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fname = f'energy_before_fnd_{deployment}.png'
    for d in {fig_dir, PAPER_FIG_DIR}:
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, fname), dpi=DPI)
    plt.close()
    return fname


def generate_all(results_dir=RESULTS_DIR):
    runs = load_runs(results_dir)
    if not runs:
        print('No runs found; nothing to plot.')
        return
    fig_dir = os.path.join(results_dir, 'figures')
    written = []
    for deployment in SCENARIO_TITLE:
        if residual_energy_before_fnd(runs, deployment, fig_dir):
            written.append(deployment)
    print(f'Wrote {len(written)} residual-energy figures to {fig_dir}')
    print(f'Mirrored into {PAPER_FIG_DIR}')


if __name__ == '__main__':
    generate_all()
