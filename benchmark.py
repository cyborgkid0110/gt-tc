"""Parallel multi-deployment benchmark sweep.

Runs every (algorithm, deployment, seed) combination in independent worker
processes, persisting one JSON per run under results/runs/. summary.csv is
rebuilt from those JSON files after the pool drains. Re-running skips existing
run files (resumable).

Run:  conda run -n base python benchmark.py
"""
import contextlib
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

ALGOS = [
    'GT2', 'LEACH', 'GTFR', 'DIA', 'MIA', 'TCLE',
    'EFTCG-1', 'EFTCG-2', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA',
    # 'EE-TCM',
]
# Each scenario is a frozen-CSV tag; the sweep loads scenarios/gen/<tag>_s<seed>.csv
# (generate them with scenarios.freeze_scenarios / scenarios.make_coverage_scenario
# first). The tag becomes the 'deployment' column in the metrics, so plots and
# summary_by_scenario.csv group by it automatically. Node count is parsed from the
# tag's _n<N> token.
SCENARIOS = [
    'uniform_n200',
    'gaussian_n100',
    'cov_free_n40_r90',
    'cov_free_n60_r60',
    'cov_obs_n40_r90',
    'cov_obs_n60_r60',
]
# 50 seeds per scenario for the box-plot / CI analysis. A superset of the
# original [1..9, 42], so earlier runs are reused. The scenario freezers
# (scenarios.freeze_scenarios / scenarios.freeze_coverage_scenarios) import
# this list, so frozen CSVs and the sweep stay in sync.
SEEDS = list(range(1, 51))
NUM_NODES = 200          # fallback when a tag has no _n<N> token
SCENARIO_DIR = os.path.join('scenarios', 'gen')
RESULTS_DIR = 'results'
WORKERS = 96
MAX_ROUNDS = 50000
# Per-scenario GT2 hyperparameters picked by gt2_grid_search.py. Tags absent
# from this file (or all tags, if the search has not been run) use
# config/gt2.yaml.
GT2_BEST_CSV = os.path.join('results', 'gridsearch', 'best_per_scenario.csv')
GT2_PARAM_KEYS = ('payoff', 'alpha', 'beta', 'mu', 'hop_max')

SUMMARY_FIELDS = [
    'algo', 'deployment', 'seed', 'num_nodes',
    'fnd', 'hnd', 'lnd',
    'total_delivered', 'total_generated', 'mean_pdr', 'cumulative_pdr',
    'energy_drained', 'energy_per_packet', 'mean_energy_std',
    'mean_avg_hop', 'mean_avg_tx_power',
]


def _disable_plotting():
    """No-op the algorithms' visualisation calls during a sweep.

    Every algorithm plots when ``t % plot_period == 0`` — which is true at t=0
    even with a huge plot_period — then calls ``plt.show()``. Under the Agg
    backend that emits warnings, wastes a render per run, and leaks figures
    (plot.py never closes them). Each algo imported the plot functions by name
    (``from plot import directional_wsn_plot``), so replace those bound names
    (and the plot module's originals) with no-ops. Must be called after
    ``main`` has imported the algorithm modules.
    """
    import sys
    noop = lambda *args, **kwargs: None
    plot_fns = ('directional_wsn_plot', 'cluster_head_probability_plot',
                'tx_power_plot')
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == 'plot' or mod_name.startswith('algos'):
            for name in plot_fns:
                if hasattr(mod, name):
                    setattr(mod, name, noop)


def _apply_gt2_override(algo_obj, net, params):
    """Override GT2's tuned knobs after construction (mirrors GT2._load_config).

    payoff lives on the algo (clustering game); alpha/beta/mu/hop_max are also
    mirrored onto the network model (power-control game / f_pr local subgraph).
    """
    algo_obj.payoff = float(params['payoff'])
    algo_obj.alpha = net.alpha = float(params['alpha'])
    algo_obj.beta = net.beta = float(params['beta'])
    algo_obj.mu = net.mu = float(params['mu'])
    # float() first: CSV-loaded values are strings, possibly '3.0'.
    algo_obj.hop_max = net.hop_max = int(float(params['hop_max']))


def load_gt2_overrides(path=GT2_BEST_CSV):
    """{tag: {payoff, alpha, beta, mu, hop_max}} from the grid search's best CSV.

    Returns {} when the file does not exist. Values are kept as the CSV
    strings; `_apply_gt2_override` casts them.
    """
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {row['deployment']: {k: row[k] for k in GT2_PARAM_KEYS}
                for row in csv.DictReader(f)}


GT2_OVERRIDES = load_gt2_overrides()


def scenario_csv(tag, seed):
    """Frozen-CSV path for a scenario tag and seed."""
    return os.path.join(SCENARIO_DIR, f'{tag}_s{seed}.csv')


def nodes_for(tag):
    """Node count parsed from a tag's _n<N> token (fallback NUM_NODES)."""
    m = re.search(r'_n(\d+)', tag)
    return int(m.group(1)) if m else NUM_NODES


def run_one(algo, tag, seed, results_dir, max_rounds=MAX_ROUNDS):
    """Run a single (algo, scenario tag, seed) and write its JSON. Resumable.

    Returns the output path. Skips (and returns the path) if it already exists.
    Loads the frozen scenario scenarios/gen/<tag>_s<seed>.csv (must exist).
    Picklable top-level function for ProcessPoolExecutor.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')
    runs_dir = os.path.join(results_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    out = os.path.join(runs_dir, f'{algo}_{tag}_{seed}.json')
    if os.path.exists(out):
        return out

    frozen = scenario_csv(tag, seed)
    if not os.path.exists(frozen):
        raise FileNotFoundError(
            f"missing frozen scenario {frozen}; generate it first "
            f"(scenarios.freeze_scenarios / scenarios.make_coverage_scenario)")

    from main import build_network, make_algo
    _disable_plotting()

    log_path = os.path.join(runs_dir, f'{algo}_{tag}_{seed}.log')
    with open(log_path, 'w') as lf, contextlib.redirect_stdout(lf):
        net = build_network(scenario=frozen)
        algo_obj = make_algo(
            algo, net, dict(max_rounds=max_rounds, plot_period=10 ** 9))
        if algo == 'GT2' and tag in GT2_OVERRIDES:
            _apply_gt2_override(algo_obj, net, GT2_OVERRIDES[tag])
        algo_obj.run()
        payload = {
            'algo': algo,
            'deployment': tag,
            'seed': seed,
            'num_nodes': nodes_for(tag),
            'summary': algo_obj.metrics.summary(),
            'time_series': algo_obj.metrics.time_series(),
        }

    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, out)
    return out


def build_summary_csv(results_dir):
    """Rebuild summary.csv from all per-run JSON files (one row per run)."""
    runs_dir = os.path.join(results_dir, 'runs')
    rows = []
    for fn in sorted(os.listdir(runs_dir)):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(runs_dir, fn)
        try:
            with open(path) as f:
                p = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'skipping unreadable run file {fn}: {e!r}')
            continue
        row = {'algo': p['algo'], 'deployment': p['deployment'],
               'seed': p['seed'], 'num_nodes': p['num_nodes']}
        row.update(p['summary'])
        rows.append(row)

    csv_path = os.path.join(results_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return csv_path


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


def missing_scenarios(tags=SCENARIOS, seeds=SEEDS):
    """Frozen scenario CSVs the sweep needs but that do not exist yet."""
    return [scenario_csv(tag, s) for tag, s in product(tags, seeds)
            if not os.path.exists(scenario_csv(tag, s))]


def main():
    missing = missing_scenarios()
    if missing:
        print(f'{len(missing)} frozen scenario CSV(s) missing, e.g.:')
        for path in missing[:10]:
            print(f'  {path}')
        print('Generate them first:\n'
              '  conda run -n base python -m scenarios.freeze_scenarios\n'
              '  conda run -n base python -m scenarios.freeze_coverage_scenarios')
        raise SystemExit(1)

    if 'GT2' in ALGOS:
        tuned = [t for t in SCENARIOS if t in GT2_OVERRIDES]
        print(f'GT2 tuned params from {GT2_BEST_CSV}: '
              f'{len(tuned)}/{len(SCENARIOS)} scenarios'
              + ('' if len(tuned) == len(SCENARIOS)
                 else ' (others use config/gt2.yaml)'))
        for t in tuned:
            print(f'  {t}: {GT2_OVERRIDES[t]}')

    os.makedirs(os.path.join(RESULTS_DIR, 'runs'), exist_ok=True)
    tasks = list(product(ALGOS, SCENARIOS, SEEDS))
    total = len(tasks)
    print(f'Dispatching {total} runs across {WORKERS} workers...')

    done = 0
    ex = ProcessPoolExecutor(max_workers=WORKERS)
    try:
        futs = {ex.submit(run_one, a, tag, s, RESULTS_DIR): (a, tag, s)
                for a, tag, s in tasks}
        for fut in as_completed(futs):
            a, tag, s = futs[fut]
            try:
                fut.result()
                done += 1
                print(f'[{done}/{total}] done: {a} / {tag} / seed {s}')
            except Exception as e:
                print(f'FAILED: {a} / {tag} / seed {s}: {e!r}')
    except KeyboardInterrupt:
        print('\nInterrupted — killing all worker processes...')
        _kill_pool(ex)
        raise SystemExit(130)
    else:
        ex.shutdown()

    path = build_summary_csv(RESULTS_DIR)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
