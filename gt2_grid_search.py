"""Grid-search hyperparameter tuning for GT2, per deployment scenario.

Sweeps GT2's four free hyperparameters — ``payoff`` (ρ, clustering game) and
``alpha`` / ``beta`` / ``mu`` (power-control game) — over a configurable grid,
for every deployment scenario, and reports the best combination per scenario
against a chosen objective metric.

Design mirrors ``benchmark.py``:
  * Same scenarios as the benchmark: each is a frozen-CSV tag loaded from
    scenarios/gen/<tag>_s<seed>.csv, so tuning happens on exactly the topologies
    the benchmark evaluates (keep SCENARIOS / SEEDS in sync with benchmark.py).
  * One task per (scenario, param-combo); each task runs all SEEDS and
    aggregates, so the objective is a mean over seeds (mean ± std).
  * Embarrassingly parallel across worker **processes**.
  * Resumable: a finished combo's JSON under results/gridsearch/runs/ is skipped.
  * After the pool drains, summary.csv (all combos) and best_per_scenario.csv
    are rebuilt from the per-combo JSON.

The shared energy model and the deployment topologies are held fixed (only GT2's
own config knobs are tuned) so the search stays inside the fairness envelope
described in docs/benchmark.md.

Run:  conda run -n base python gt2_grid_search.py
"""
import contextlib
import csv
import itertools
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ---------------------------------------------------------------------------- #
#  Search configuration (edit here)
# ---------------------------------------------------------------------------- #
# Candidate values per hyperparameter, calibrated to the current energy model
# (E0 = 0.5 J, P in [0.078, 0.653] W; see main.py).
#
# payoff (rho) must exceed the CH role cost c_ch (~1e-4 .. 5e-2 J) before any
# node volunteers; below ~3e-3 GT2 elects ~0 CHs and every round falls back to
# _maintenance_no_cluster, so the power-control knobs are never exercised. The
# range below elects roughly 2-30 % of the nodes as CHs across the scenarios.
#
# alpha / beta / mu enter GT2 only through u = beta*|V'| - alpha*var(E) - C(p)/mu
# and only utility comparisons matter, so the decisions depend solely on
# alpha*mu and beta*mu (verified: (3, 0.9, 0.09) and (30, 9, 0.009) give
# identical runs). mu is therefore fixed and alpha / beta alone are swept.
#   beta*mu vs one power step (C(p_step)*mu ~ 0.023): below it a node trades a
#     k-hop vertex for a lower power level; above it the node keeps the vertex.
#   alpha*mu: dropping one vertex shifts var(E_res) by ~1e-4 .. 2e-3 J^2, so the
#     energy-balance term only competes with a power step for alpha*mu ~ 10-100
#     (alpha = 0 switches the term off as a baseline).
#
# Refined around the coarse search (payoff 5e-3..1.2e-1, beta 0.03..3, alpha
# 0..1000, 8 seeds). FND fell monotonically with payoff, so every scenario's
# best sat at the grid's lower edge (5e-3); payoff = 0 (no clustering) is far
# worse (FND ~350-560 vs ~1400-2800), so the optimum lies in (0, 1e-2). beta
# won at its lower edge (0.03) in 5/6 scenarios, and alpha 0 vs 1000 moved FND
# less than the seed noise; only its extremes are kept.
GRID = {
    'payoff': [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 7e-3, 1e-2, 1.4e-2],  # rho: CH benefit -> CH count
    'alpha':  [0, 1000],                         # energy-balance weight (f_e); no measurable effect
    'beta':   [0, 0.01, 0.03, 0.06, 0.1],        # connectivity weight (f_pr); beta*mu 0-0.01
    'mu':     [0.1],                             # fixed: only alpha*mu, beta*mu matter
    'hop_max':[2, 3, 4, 5],                      # k-hop radius for f_pr / local subgraph
}

# Scenarios match benchmark.py: each tag is a frozen-CSV scenario loaded from
# scenarios/gen/<tag>_s<seed>.csv (generate first with scenarios.freeze_scenarios
# / scenarios.make_coverage_scenario). The tag is the 'deployment' column; node
# count is parsed from its _n<N> token. Keep these in sync with benchmark.py so
# tuning happens on the same envelope the benchmark evaluates.
SCENARIOS = [
    'uniform_n200',
    'gaussian_n100',
    'cov_free_n40_r90',
    'cov_free_n60_r60',
    'cov_obs_n40_r90',
    'cov_obs_n60_r60',
]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]    # subset of benchmark seeds — keeps tuning fast
NUM_NODES = 200               # fallback when a tag has no _n<N> token
SCENARIO_DIR = os.path.join('scenarios', 'gen')
MAX_ROUNDS = 50000            # match benchmark.py's round cap
WORKERS = os.cpu_count()
RESULTS_DIR = os.path.join('results', 'gridsearch')

# Objective: which summary metric to optimise, and the direction.
OBJECTIVE = 'fnd'             # 'fnd' | 'hnd' | 'lnd' | 'total_delivered' |
                              # 'cumulative_pdr' | 'energy_per_packet' | ...
_MINIMIZE = {'energy_per_packet', 'mean_energy_std', 'energy_drained'}

# Summary scalars carried into the CSVs (mean across seeds).
_REPORT_METRICS = [
    'fnd', 'hnd', 'lnd', 'total_delivered',
    'cumulative_pdr', 'energy_drained', 'energy_per_packet', 'mean_energy_std',
]


def _disable_plotting():
    """No-op GT2's visualisation calls (it plots at t=0 even with huge period)."""
    import sys
    noop = lambda *a, **k: None
    plot_fns = ('directional_wsn_plot', 'cluster_head_probability_plot',
                'tx_power_plot')
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == 'plot' or mod_name.startswith('algos'):
            for name in plot_fns:
                if hasattr(mod, name):
                    setattr(mod, name, noop)


def _combo_id(params):
    """Stable, filename-safe identifier for a parameter combination."""
    return '_'.join(f'{k}{params[k]:g}' for k in sorted(params))


def scenario_csv(tag, seed):
    """Frozen-CSV path for a scenario tag and seed (mirrors benchmark.py)."""
    return os.path.join(SCENARIO_DIR, f'{tag}_s{seed}.csv')


def nodes_for(tag):
    """Node count parsed from a tag's _n<N> token (fallback NUM_NODES)."""
    m = re.search(r'_n(\d+)', tag)
    return int(m.group(1)) if m else NUM_NODES


def _run_one_sim(tag, params, seed):
    """Build a GT2 with overridden hyperparameters and run it to death.

    Loads the frozen scenario scenarios/gen/<tag>_s<seed>.csv (same source as
    benchmark.py). Seeds the global ``random`` and legacy numpy RNG (GT2's
    clustering game and the energy model both draw from ``random``) so the
    stochastic parts of a run are reproducible across param combos. Returns the
    metrics summary dict.
    """
    from main import build_network
    from algos.gt2 import GT2

    random.seed(seed)
    np.random.seed(seed)

    frozen = scenario_csv(tag, seed)
    if not os.path.exists(frozen):
        raise FileNotFoundError(
            f"missing frozen scenario {frozen}; generate it first "
            f"(scenarios.freeze_scenarios / scenarios.make_coverage_scenario)")

    net = build_network(scenario=frozen)
    algo = GT2(net, config_path='config/gt2.yaml',
               max_rounds=MAX_ROUNDS, plot_period=10 ** 9)

    # Override the free knobs. payoff lives on the algo (clustering game);
    # alpha/beta/mu/hop_max are mirrored onto the network model (power-control
    # game / f_pr local subgraph), matching GT2._load_config.
    algo.payoff = params['payoff']
    algo.alpha = net.alpha = params['alpha']
    algo.beta = net.beta = params['beta']
    algo.mu = net.mu = params['mu']
    algo.hop_max = net.hop_max = int(params['hop_max'])

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        algo.run()
    return algo.metrics.summary()


def run_combo(tag, params, results_dir=RESULTS_DIR):
    """Run all SEEDS for one (scenario, combo), aggregate, persist JSON. Resumable.

    Picklable top-level function for ProcessPoolExecutor. Returns the JSON path.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')
    runs_dir = os.path.join(results_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    out = os.path.join(runs_dir, f'{tag}__{_combo_id(params)}.json')
    if os.path.exists(out):
        return out

    _disable_plotting()

    per_seed = [_run_one_sim(tag, params, s) for s in SEEDS]

    # Aggregate: mean of each reported metric across seeds (None -> skipped).
    agg = {}
    for m in _REPORT_METRICS:
        vals = [r[m] for r in per_seed if r.get(m) is not None]
        agg[m] = float(np.mean(vals)) if vals else None
        agg[f'{m}_std'] = float(np.std(vals)) if vals else None

    obj_vals = [r[OBJECTIVE] for r in per_seed if r.get(OBJECTIVE) is not None]
    objective = float(np.mean(obj_vals)) if obj_vals else None

    payload = {
        'deployment': tag,
        'params': params,
        'seeds': SEEDS,
        'objective_metric': OBJECTIVE,
        'objective': objective,
        'metrics': agg,
    }
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, out)
    return out


def _load_records(results_dir):
    runs_dir = os.path.join(results_dir, 'runs')
    recs = []
    for fn in sorted(os.listdir(runs_dir)):
        if fn.endswith('.json'):
            with open(os.path.join(runs_dir, fn)) as f:
                recs.append(json.load(f))
    return recs


def build_summaries(results_dir=RESULTS_DIR):
    """Rebuild summary.csv (all combos) and best_per_scenario.csv from JSON."""
    recs = _load_records(results_dir)

    param_keys = sorted(GRID)
    fields = (['deployment'] + param_keys
              + ['objective_metric', 'objective']
              + _REPORT_METRICS)

    def _row(r):
        row = {'deployment': r['deployment']}
        row.update({k: r['params'][k] for k in param_keys})
        row['objective_metric'] = r['objective_metric']
        row['objective'] = r['objective']
        row.update({m: r['metrics'].get(m) for m in _REPORT_METRICS})
        return row

    all_path = os.path.join(results_dir, 'summary.csv')
    with open(all_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in sorted(recs, key=lambda r: (r['deployment'],
                                             tuple(r['params'][k] for k in param_keys))):
            w.writerow(_row(r))

    # Best combo per scenario.
    minimize = OBJECTIVE in _MINIMIZE
    best = {}
    for r in recs:
        if r['objective'] is None:
            continue
        cur = best.get(r['deployment'])
        if cur is None or (r['objective'] < cur['objective']) == minimize:
            best[r['deployment']] = r

    best_path = os.path.join(results_dir, 'best_per_scenario.csv')
    with open(best_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for dep in sorted(best):
            w.writerow(_row(best[dep]))

    print(f'\nBest GT2 hyperparameters per scenario '
          f'({"min" if minimize else "max"} {OBJECTIVE}):')
    for dep in sorted(best):
        r = best[dep]
        ps = ', '.join(f'{k}={r["params"][k]:g}' for k in param_keys)
        print(f'  {dep:18} {OBJECTIVE}={r["objective"]:.1f}  ({ps})')
    return all_path, best_path


def _kill_pool(ex):
    """Forcibly terminate every worker of a ProcessPoolExecutor.

    A plain Ctrl-C is not enough: the executor's ``__exit__`` runs
    ``shutdown(wait=True)``, which blocks until the in-flight CPU-bound runs
    finish, and the workers catch the SIGINT inside ``_process_worker`` and
    loop back for more work instead of exiting. The parent then dies on a
    second Ctrl-C, orphaning the workers (reparented to init, still burning
    CPU). SIGKILL the workers directly so they cannot ignore it, then drop any
    queued futures without waiting.
    """
    for p in list(getattr(ex, '_processes', {}).values()):
        p.kill()
    ex.shutdown(wait=False, cancel_futures=True)


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'runs'), exist_ok=True)
    param_keys = sorted(GRID)
    combos = [dict(zip(param_keys, vals))
              for vals in itertools.product(*(GRID[k] for k in param_keys))]
    tasks = [(tag, c) for tag in SCENARIOS for c in combos]
    total = len(tasks)
    print(f'GT2 grid search: {len(combos)} combos x {len(SCENARIOS)} '
          f'scenarios = {total} tasks ({len(SEEDS)} seeds each), '
          f'objective={OBJECTIVE}, {WORKERS} workers.')

    done = 0
    ex = ProcessPoolExecutor(max_workers=WORKERS)
    try:
        futs = {ex.submit(run_combo, tag, c): (tag, c) for tag, c in tasks}
        for fut in as_completed(futs):
            tag, c = futs[fut]
            try:
                fut.result()
                done += 1
                print(f'[{done}/{total}] {tag} / {_combo_id(c)}')
            except Exception as e:
                print(f'FAILED: {tag} / {_combo_id(c)}: {e!r}')
    except KeyboardInterrupt:
        print('\nInterrupted — killing all worker processes...')
        _kill_pool(ex)
        raise SystemExit(130)
    else:
        ex.shutdown()

    build_summaries(RESULTS_DIR)


if __name__ == '__main__':
    main()
