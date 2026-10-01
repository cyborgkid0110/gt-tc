"""Generate LaTeX table bodies for the paper's result section from results/.

Three tables (per GT2_paper/sec4_simulation.tex):
  - avg_hop at selected rounds, one table per scenario (mean +/- std over seeds)
  - avg_tx_power (uW) at selected rounds, one table per scenario
  - energy-per-packet (J) by scenario (single table, mean over seeds)

Round cells aggregate over the 10 seeds: a seed contributes its value at that
exact round only if its network still has a node alive then; '--' when no seed
survives to that round. Run: conda run -n base python paper_tables.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

RESULTS = 'results'
ALGOS = ['GT2', 'LEACH', 'GTFR', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA',
         'DIA', 'MIA', 'TCLE', 'EFTCG-1', 'EFTCG-2']
ALGO_TEX = {'SCA-LEVY': "SCA-L\\'evy"}
SCENARIOS = ['uniform_n200', 'gaussian_n100',
             'cov_free_n40_r90', 'cov_free_n60_r60',
             'cov_obs_n40_r90', 'cov_obs_n60_r60']
ROUNDS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]


def load():
    """runs[(algo, scenario)] -> list of time_series dicts (one per seed)."""
    runs = defaultdict(list)
    for fn in glob.glob(os.path.join(RESULTS, 'runs', '*.json')):
        d = json.load(open(fn))
        runs[(d['algo'], d['deployment'])].append(d['time_series'])
    return runs


def value_at(ts, key, target):
    """ts[key] at the exact round == target, or None if not recorded."""
    rounds = ts['rounds']
    if not rounds:
        return None
    idx = target - rounds[0]          # rounds are consecutive
    if 0 <= idx < len(rounds) and rounds[idx] == target:
        return ts[key][idx]
    return None


def cell(values, scale, fmt):
    """mean +/- std of a list, scaled; '--' if empty."""
    if not values:
        return '--'
    arr = np.asarray(values, float) * scale
    return fmt.format(arr.mean(), arr.std())


def round_table(runs, scenario, key, scale, fmt):
    lines = []
    for algo in ALGOS:
        series = runs.get((algo, scenario), [])
        cells = []
        for r in ROUNDS:
            vals = [v for ts in series
                    if (v := value_at(ts, key, r)) is not None]
            cells.append(cell(vals, scale, fmt))
        name = ALGO_TEX.get(algo, algo)
        lines.append(f"{name:12s} & " + ' & '.join(cells) + r' \\')
    return '\n'.join(lines)


def epp_table():
    import csv
    cells = defaultdict(dict)   # cells[algo][scenario] = mean J
    for row in csv.DictReader(open(os.path.join(RESULTS, 'summary_by_scenario.csv'))):
        v = row['energy_per_packet_mean']
        if v != '':
            cells[row['algo']][row['deployment']] = float(v)
    lines = []
    for algo in ALGOS:
        vals = []
        for s in SCENARIOS:
            x = cells.get(algo, {}).get(s)
            if x is None:
                vals.append('--')
            else:
                m, e = f'{x:.2e}'.split('e')
                vals.append(f'${m}\\times10^{{{int(e)}}}$')
        name = ALGO_TEX.get(algo, algo)
        lines.append(f"{name:12s} & " + ' & '.join(vals) + r' \\')
    return '\n'.join(lines)


SCEN_TEX = {s: s.replace('_', r'\_') for s in SCENARIOS}

ROUND_FLOAT = r"""\begin{{table}}[htbp]
\centering
\footnotesize
\caption{{{caption}}}
\label{{tab:{label}}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabularx}}{{\textwidth}}{{l *{{8}}{{>{{\centering\arraybackslash}}X}}}}
\hline
\hline
\textbf{{Algorithm}} & \textbf{{5k}} & \textbf{{10k}} & \textbf{{15k}} & \textbf{{20k}} & \textbf{{25k}} & \textbf{{30k}} & \textbf{{35k}} & \textbf{{40k}} \\
\hline
{body}
\hline
\hline
\end{{tabularx}}
\end{{table}}"""


def round_float(runs, scenario, key, scale, fmt, metric, label_prefix):
    body = round_table(runs, scenario, key, scale, fmt)
    caption = f"{metric} for the {SCEN_TEX[scenario]} scenario " \
              r"(mean $\pm$ std over 10 seeds; -- = no node alive at that round)."
    return ROUND_FLOAT.format(caption=caption,
                              label=f'{label_prefix}_{scenario}', body=body)


def main():
    runs = load()
    print('% ==================== AVG HOP TABLES ====================')
    for s in SCENARIOS:
        print(round_float(
            runs, s, 'avg_hop', 1.0, r'{:.2f} $\pm$ {:.2f}',
            'Average hop count to base station at selected rounds', 'avg_hop'))
        print()
    print('% ==================== AVG TX POWER TABLES (uW) ============')
    for s in SCENARIOS:
        print(round_float(
            runs, s, 'avg_tx_power', 1e6, r'{:.0f} $\pm$ {:.0f}',
            r'Average transmit power per node ($\mu$W) at selected rounds',
            'avg_tx_power'))
        print()
    print('% ==================== ENERGY PER PACKET (J) ===============')
    print(epp_table())


if __name__ == '__main__':
    main()
