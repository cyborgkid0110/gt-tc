# gt-tc — Game-Theoretic Topology Control for WSNs

Benchmarks the **GT2** two-stage game-theoretic protocol against standard WSN
algorithms (LEACH, GTFR, DIA/MIA, TCLE, EFTCG, FL-LEACH-PSO, SCA-Lévy, FC-CRA,
EE-TCM) on a shared physical/energy model. See `CLAUDE.md` for architecture and
`docs/benchmark.md` for the full benchmark documentation.

## Setup

Uses Anaconda. Activate `base` (which has the required packages) before running:

```bash
conda activate base                       # numpy scipy matplotlib networkx pyyaml shapely
```

`shapely` is only needed for the coverage-deployment generator (see
[Custom coverage scenarios](#custom-coverage-scenarios)); everything else runs
without it.

<details>
<summary>Alternative: plain virtualenv</summary>

```bash
python -m venv venv
source venv/bin/activate                  # Windows: ./venv/Scripts/activate
pip install numpy scipy matplotlib networkx pyyaml shapely
```
</details>

## Running a single simulation (`main.py`)

```bash
python main.py                            # default: GT2, poisson deployment
MPLBACKEND=Agg python main.py             # no interactive plot windows
```

CLI flags (a run is fully reproducible from `deployment` + `num-nodes` + `seed`):

```bash
python main.py --algo LEACH --deployment uniform --num-nodes 200 --seed 1
```

| Flag | Default | Choices |
|------|---------|---------|
| `--algo` | `GT2` | `GT2`, `LEACH`, `GTFR`, `DIA`, `MIA`, `TCLE`, `EFTCG-1`, `EFTCG-2`, `FL-LEACH-PSO`, `SCA-LEVY`, `FC-CRA`, `EE-TCM` |
| `--deployment` | `poisson` | `poisson`, `uniform`, `grid`, `gaussian`, `edge` |
| `--num-nodes` | `200` | any positive integer |
| `--seed` | `42` | any integer |

Global constants (energy, radio, area, `MAX_ROUNDS`) live in `main.py`;
per-algorithm hyperparameters live in `config/<algo>.yaml`.

## Frozen scenarios

A deployment can be **frozen to a CSV** and replayed instead of regenerated, so a
benchmark always runs on identical node positions. A scenario file stores each
node's `x, y, vpre` plus the base-station position (`bs_x`, `bs_y`) and
provenance in `#` header comments (see `scenarios/store.py`). All scenario code
lives in the `scenarios/` package; generated CSVs land in `scenarios/gen/`.

```bash
conda run -n base python -m scenarios.freeze_scenarios   # -> scenarios/gen/<deployment>_n<N>_s<seed>.csv
python main.py --scenario scenarios/gen/uniform_n200_s7.csv   # run from a frozen file
```

`scenarios.freeze_scenarios` snapshots the benchmark's full `(deployment, seed)`
grid (idempotent — existing files are skipped). Once `scenarios/gen/*.csv` exist,
`benchmark.py` automatically prefers the frozen file for a given
`(deployment, num_nodes, seed)` and falls back to live generation when none is
present. `--scenario` overrides `--deployment`/`--num-nodes`/`--seed`.

## Custom coverage scenarios

Place nodes to **maximise area coverage** over a region that may contain
obstacles or winding paths, with the base station anywhere off-obstacle. A region
is authored as a YAML definition under `scenarios/defs/` (polygons via
[`shapely`](#setup)):

```yaml
# scenarios/defs/example.yaml
name: example
area: 250                 # square bounds [-area, area]^2
num_nodes: 60
coverage_radius: 45       # design-time coverage disk radius (m)
seed: 7
bs: [-200.0, -200.0]      # base station (must be outside obstacles / on paths)
obstacles:                # list of polygons [[x, y], ...]
  - [[-60, -60], [60, -60], [60, 60], [-60, 60]]
paths: []                 # optional; if non-empty, nodes deploy ONLY on the
                          #   buffered paths, e.g. {coords: [...], width: 30}
```

```bash
conda run -n base python -m scenarios.make_coverage_scenario --def scenarios/defs/example.yaml
# -> scenarios/gen/example.csv, then:
python main.py --algo LEACH --scenario scenarios/gen/example.csv
```

Visualise a scenario (deployable area, obstacles, paths, coverage disks, BS):

```bash
conda run -n base python plot_deployments.py --def scenarios/defs/example.yaml --show-links
# or, for any frozen CSV (nodes + BS only): --scenario scenarios/gen/uniform_n200_s7.csv
# -> docs/figures/scenario_<name>.png
```

Placement uses a projected particle-swarm optimiser (`coverage_deploy.py`, knobs
in `config/coverage.yaml`) that maximises coverage while penalising
disconnection, followed by a hard connectivity repair so every layout is
reachable from the BS at full transmit power. The `coverage_radius` is a
design-time spacing knob only — the benchmark's physics comm range (the shared
energy model) is unchanged, keeping coverage scenarios comparable to the random
ones.

## Running the benchmark sweep

Run every `(algorithm, deployment, seed)` combination in parallel, persist
metrics, then render figures:

```bash
conda run -n base python benchmark.py        # -> results/runs/*.json + results/summary.csv
conda run -n base python plot_benchmark.py   # -> results/figures/*.png + results/summary_by_scenario.csv
conda run -n base python plot_deployments.py # -> docs/figures/deployments.png (scenario maps)
```

- **Sweep config** (algorithms, deployments, seeds, worker count) is in the
  constants at the top of `benchmark.py`.
- **Resumable:** existing `results/runs/*.json` are skipped, so a crashed or
  partial sweep is resumed by re-running. Safe to launch in the background.
- `plot_benchmark.py` never re-runs the sweep — it only reads persisted JSON, so
  iterating on figures is cheap. It emits both pooled bar charts and
  per-scenario breakdowns.

Full details — scenarios, metric definitions, and how each metric is
calculated — are in [`docs/benchmark.md`](docs/benchmark.md).

## Tuning GT2 hyperparameters

Grid-search GT2's four free knobs (`payoff`, `alpha`, `beta`, `mu`) per
deployment scenario and report the best combination per scenario:

```bash
conda run -n base python gt2_grid_search.py   # -> results/gridsearch/{summary,best_per_scenario}.csv
```

The grid, seeds, optimisation objective (default: maximise HND), and worker
count are constants at the top of `gt2_grid_search.py`. Runs are parallel and
resumable (finished combos under `results/gridsearch/runs/` are skipped). Only
GT2's own config knobs are tuned — the shared energy model, deployment, and node
count stay fixed, keeping the search inside the benchmark's fairness envelope.
