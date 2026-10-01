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
is authored as a YAML definition under `scenarios/defs/` (polygons and circles via
[`shapely`](#setup)):

```yaml
# scenarios/defs/example.yaml
name: example
area: 250                 # square bounds [-area, area]^2
num_nodes: 60
coverage_radius: 45       # design-time coverage disk radius (m)
seed: 7                   # scalar -> one CSV; or a list [7, 8, 9] -> one CSV per seed
bs: [-200.0, -200.0]      # base station (must be outside obstacles / on paths)
obstacles:                # optional; list of polygons [[x, y], ...]
  - [[-60, -60], [60, -60], [60, 60], [-60, 60]]
circles:                  # optional; list of circular obstacles [cx, cy, r]
  - [0.0, 0.0, 80.0]
paths: []                 # optional; if non-empty, nodes deploy ONLY on the
                          #   buffered paths, e.g. {coords: [...], width: 30}
```

`obstacles` (polygons) and `circles` ([cx, cy, r] discs) are both optional and
combine; omit a key for none of that kind.

```bash
conda run -n base python -m scenarios.make_coverage_scenario --def scenarios/defs/example.yaml
# -> scenarios/gen/example.csv, then:
python main.py --algo LEACH --scenario scenarios/gen/example.csv

# --name overrides the YAML 'name' (output filename):
conda run -n base python -m scenarios.make_coverage_scenario --def scenarios/defs/example.yaml --name myrun
# -> scenarios/gen/myrun.csv
```

Visualise a scenario (deployable area, obstacles, paths, coverage disks, BS):

```bash
conda run -n base python plot_deployments.py --def scenarios/defs/example.yaml --show-links
# or, for any frozen CSV (nodes + BS only): --scenario scenarios/gen/uniform_n200_s7.csv
# -> docs/figures/scenario_<name>.png
conda run -n base python plot_deployments.py --scenario scenarios/gen/uniform_n200_s7.csv
```

* Placement uses a projected PSO (`coverage_deploy.py`; parameters in `config/coverage.yaml`) to maximise coverage deployment with connectivity constraints.
* `coverage_radius` only controls deployment spacing during scenario generation.
* `make_coverage_scenario` generates one CSV per seed:
  * Single seed (`seed: 7`) → `<name>.csv`
  * Multiple seeds (`seed: [7, 8, 9]`) → `<name>_s7.csv`, `<name>_s8.csv`, `<name>_s9.csv`
* Each seed produces an independent PSO-generated layout; multi-seed generation runs in parallel.
* `--name` overrides the scenario definition's `name` and sets the output filename prefix.
* `--workers N` limits the parallel worker pool size (default: `os.cpu_count()`).


For the standard multi-seed evaluation set, `scenarios.freeze_coverage_scenarios` snapshots
the four built-in coverage configurations — two free-space and two with a
central circular obstacle — over 10 seeds each, in parallel and idempotently:

```bash
conda run -n base python -m scenarios.freeze_coverage_scenarios
# -> scenarios/gen/cov_{free,obs}_n<N>_r<R>_s<seed>.csv   (40 files)
```

The configurations (node count, coverage radius, obstacle, BS) are defined in
`COVERAGE_CONFIGS` at the top of that module; edit them there to change the set.

## Running the benchmark sweep

Run every `(algorithm, scenario, seed)` combination in parallel, persist
metrics, then render figures:

```bash
conda run -n base python benchmark.py        # -> results/runs/*.json + results/summary.csv
conda run -n base python plot_benchmark.py   # -> results/figures/*.png + results/summary_by_scenario.csv
conda run -n base python plot_deployments.py # -> docs/figures/deployments.png (scenario maps)
```

- **Sweep config** (algorithms, scenarios, seeds, worker count) is in the
  constants at the top of `benchmark.py`. `SCENARIOS` is a list of **frozen-CSV
  tags**; each run loads `scenarios/gen/<tag>_s<seed>.csv`, so random and
  coverage scenarios are swept identically. The tag becomes the `deployment`
  column in the metrics, so `plot_benchmark.py` groups charts and
  `summary_by_scenario.csv` by it automatically. Node count is parsed from the
  tag's `_n<N>` token. Example default set:
  `uniform_n200`, `gaussian_n100`, `cov_{free,obs}_n{40,60}_r{90,60}`.
- **Freeze first:** every listed tag must already have its CSVs under
  `scenarios/gen/` — generate them with `scenarios.freeze_scenarios` (random)
  and `scenarios.freeze_coverage_scenarios` / `scenarios.make_coverage_scenario`
  (coverage). A run with no matching CSV fails fast (no live fallback).
- **Resumable:** existing `results/runs/*.json` are skipped, so a crashed or
  partial sweep is resumed by re-running. Safe to launch in the background.
- `plot_benchmark.py` never re-runs the sweep — it only reads persisted JSON, so
  iterating on figures is cheap. It emits both pooled bar charts and
  per-scenario breakdowns.

**Tile frozen scenarios into one figure** (`plot_deployments.py --grid`):

```bash
conda run -n base python plot_deployments.py --grid \
  scenarios/gen/uniform_n200_s1.csv scenarios/gen/gaussian_n100_s1.csv \
  scenarios/gen/cov_free_n40_r90_s1.csv scenarios/gen/cov_free_n60_r60_s1.csv \
  scenarios/gen/cov_obs_n40_r90_s1.csv scenarios/gen/cov_obs_n60_r60_s1.csv
```

- One panel per CSV, in the order given; output `docs/figures/scenarios_grid.png`.
- Layout `--rows`/`--cols` (default `2`/`3`); `--out PATH` to override the path.
- Obstacle circles are drawn for known coverage tags (`cov_obs_*`).
- `--titles "(a) Free space (N=200)" ...` overrides captions (one per CSV, in order); default is the scenario tag.

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
