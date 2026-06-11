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
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

ALGOS = [
    'GT2', 'LEACH', 'GTFR', 'DIA', 'MIA', 'TCLE',
    'EFTCG-1', 'EFTCG-2', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA', 
    # 'EE-TCM',
]
DEPLOYMENTS = ['poisson', 'uniform', 'grid', 'gaussian', 'edge']
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
NUM_NODES = 200
# Per-deployment node-count overrides; deployments not listed use NUM_NODES.
# gaussian is run at 100 nodes only (denser hotspots at lower count).
DEPLOYMENT_NODES = {'gaussian': 100}
RESULTS_DIR = 'results'
WORKERS = os.cpu_count() or 4
MAX_ROUNDS = 50000

SUMMARY_FIELDS = [
    'algo', 'deployment', 'seed', 'num_nodes',
    'fnd', 'hnd', 'lnd',
    'total_delivered', 'total_generated', 'mean_pdr', 'cumulative_pdr',
    'energy_drained', 'energy_per_packet', 'mean_energy_std',
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


def run_one(algo, deployment, seed, num_nodes, results_dir,
            max_rounds=MAX_ROUNDS):
    """Run a single (algo, deployment, seed) and write its JSON. Resumable.

    Returns the output path. Skips (and returns the path) if it already exists.
    Picklable top-level function for ProcessPoolExecutor.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')
    runs_dir = os.path.join(results_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    out = os.path.join(runs_dir, f'{algo}_{deployment}_{seed}.json')
    if os.path.exists(out):
        return out

    from main import build_network, make_algo
    from scenarios.freeze_scenarios import scenario_path
    _disable_plotting()

    log_path = os.path.join(runs_dir, f'{algo}_{deployment}_{seed}.log')
    with open(log_path, 'w') as lf, contextlib.redirect_stdout(lf):
        frozen = scenario_path(deployment, num_nodes, seed)
        if os.path.exists(frozen):
            net = build_network(scenario=frozen)
        else:
            net = build_network(deployment, num_nodes, seed)
        algo_obj = make_algo(
            algo, net, dict(max_rounds=max_rounds, plot_period=10 ** 9))
        algo_obj.run()
        payload = {
            'algo': algo,
            'deployment': deployment,
            'seed': seed,
            'num_nodes': num_nodes,
            'summary': algo_obj.metrics.summary(),
            'time_series': algo_obj.metrics.time_series(),
        }

    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, out)
    return out


def nodes_for(deployment):
    """Node count for a deployment, honouring DEPLOYMENT_NODES overrides."""
    return DEPLOYMENT_NODES.get(deployment, NUM_NODES)


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


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'runs'), exist_ok=True)
    tasks = list(product(ALGOS, DEPLOYMENTS, SEEDS))
    total = len(tasks)
    print(f'Dispatching {total} runs across {WORKERS} workers...')

    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, a, d, s, nodes_for(d), RESULTS_DIR): (a, d, s)
                for a, d, s in tasks}
        for fut in as_completed(futs):
            a, d, s = futs[fut]
            try:
                fut.result()
                done += 1
                print(f'[{done}/{total}] done: {a} / {d} / seed {s}')
            except Exception as e:
                print(f'FAILED: {a} / {d} / seed {s}: {e!r}')

    path = build_summary_csv(RESULTS_DIR)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
