# WSN Benchmark Metrics Harness — Design

**Date:** 2026-06-01
**Status:** Approved (design), pending implementation plan
**Goal:** Add a metrics-collection and benchmarking layer so GT2 can be compared
against the other implemented WSN algorithms on a consistent, fair basis.

## Motivation

The project pipeline (CLAUDE.md) reaches "Run experiments & collect metrics" and
"Produce simulation results", but no structured metrics infrastructure exists.
Current `benchmark_logs/*.txt` are free-text per-round dumps (CH count, dead
count) with no energy, delivery, or aggregation data. This design adds:

1. A canonical, well-defined set of comparison metrics.
2. A shared collection layer so every algorithm is measured identically (same
   fairness principle as the shared energy model).
3. A reproducible multi-deployment sweep runner that persists raw data.
4. Plots generated from persisted data without re-running the sweep.

## Scope decisions (from brainstorming)

- **Benchmark goal:** all four families matter — lifetime, data delivery,
  topology quality, energy efficiency.
- **Cross-family handling:** *universal core + family-specific extras.* Every
  algorithm reports a universal core; clustering and topology-control algorithms
  each add their own extras.
- **Experiment design:** *multi-deployment sweep* — every algorithm across all
  deployment scenarios × a few seeds each; report mean ± std.
- **Output:** matplotlib plots + CSV/JSON raw data. (Markdown tables not in
  scope; `summary.csv` makes them trivial to derive later.)
- **Architecture:** Approach A — shared metrics hooks in `BaseAlgorithm` +
  standalone sweep runner + standalone plotting module.

## Metric catalog

### Universal core (every algorithm reports)

| Metric | Definition | Source |
|--------|------------|--------|
| **FND** | First Node Death — first round where `alive < N` | alive-count series |
| **HND** | Half Node Death — first round where `alive ≤ N/2` | alive-count series |
| **LND** | Last Node Death — final round of the run | alive-count series |
| **Survival curve** | alive nodes vs round | per-round series |
| **Total residual energy** | Σ `e_res` over alive nodes, per round | `NetworkModel` |
| **Energy balance** | std-dev of `e_res` across alive nodes, per round (lower = better balanced) | `NetworkModel` |
| **Energy consumed/round** | drop in total residual energy between rounds | derived |
| **Packets delivered to BS** | cumulative count of alive nodes with a path to BS | `build_routing_tree()` |
| **PDR** | delivered / generated (mean across rounds and cumulative) | derived |
| **Throughput** | delivered packets per round | per-round series |
| **Energy per delivered packet** | total energy drained ÷ total delivered | derived |
| **Average hop count to BS** | mean routing-tree depth over delivered nodes, +1 for the final hop to the sink (0.0 if nothing reaches the BS) | `build_routing_tree()` |
| **Average tx power / node** | mean `s.power` over alive nodes | `NetworkModel` |

> **Update 2026-06-13:** *Average hop count to BS* and *Average tx power / node*
> were promoted to **universal-core** (computed in `record_round` for every
> algorithm). The routing tree built for *delivered* is reused for *avg hop*.
> *Average tx power* is therefore no longer a topology-family extra.

### Clustering-family extras
*(GT2, LEACH, GTFR, FL-LEACH-PSO, SCA-Lévy, FC-CRA, EE-TCM)*

| Metric | Definition |
|--------|------------|
| **CH count/round** | number of cluster heads elected each round |
| **CH load balance** | std-dev of cluster sizes (members per CH) |

### Topology-family extras
*(DIA, MIA, LDIA, TCLE, EFTCG)*

| Metric | Definition |
|--------|------------|
| **Average node degree** | mean out-degree on the directed adjacency graph |
| **Algebraic connectivity λ₂ / k-connectivity** | structural robustness (TCLE/EFTCG already compute these) |

*(Average tx power moved to the universal-core table above as of 2026-06-13.)*

## Data-delivery definition (fair, algorithm-agnostic)

- One data packet is generated per alive node per round.
- A node **delivers** iff a directed path to the BS exists on the current active
  topology. This is computed by the existing
  `NetworkModel.build_routing_tree()` — a node appears in the returned tree iff
  it (or a relay it reaches) is within range of the BS at origin.
- Therefore, per round:
  - `generated = alive_count`
  - `delivered = len(net.build_routing_tree())`
  - `pdr = delivered / generated` (guard divide-by-zero when `generated == 0`)
- PDR stays ≈ 1.0 until the topology partitions or relays die — that degradation
  is the discriminating signal. This definition does **not** rely on any
  algorithm's internal routing bookkeeping, preserving fairness.

## Architecture (Approach A)

### Components

**1. `metrics.py` — `MetricsCollector`** (one instance per run)

- State: `num_nodes`, per-round lists (`round`, `alive`, `total_energy`,
  `energy_std`, `delivered`, `generated`, plus a dict of family-extra series),
  and `initial_total_energy` captured at construction.
- `record_round(net, t, family_extras: dict)` — appends one row. Computes
  universal-core per-round values from `net`:
  - `alive` = count of `s.is_alive`
  - `total_energy` = Σ `s.e_res` for alive
  - `energy_std` = std-dev of `s.e_res` over alive (0.0 if < 2 alive)
  - `delivered` = `len(net.build_routing_tree())` (tree built once, reused below)
  - `generated` = `alive`
  - `avg_hop` = mean routing-tree `depth` over delivered nodes + 1 (0.0 if empty)
  - `avg_tx_power` = mean `s.power` over alive nodes (0.0 if none alive)
  - merges `family_extras` into the family-series dict.
- `finalize()` — derives scalar markers once the run ends:
  - `fnd` = first `round` where `alive < num_nodes` (else `None`)
  - `hnd` = first `round` where `alive ≤ num_nodes/2` (else `None`)
  - `lnd` = last recorded `round`
  - `total_delivered` = Σ delivered; `total_generated` = Σ generated
  - `mean_pdr` = mean of per-round `delivered/generated`
  - `cumulative_pdr` = `total_delivered / total_generated`
  - `energy_drained` = `initial_total_energy − final total_energy`
  - `energy_per_packet` = `energy_drained / total_delivered` (guard 0)
  - `mean_energy_std` = mean of per-round `energy_std`
  - `mean_avg_hop` = mean of per-round `avg_hop`
  - `mean_avg_tx_power` = mean of per-round `avg_tx_power`
- `summary() -> dict` — flat dict of the scalars above (CSV row).
- `time_series() -> dict` — the per-round lists (JSON payload).

**2. `BaseAlgorithm` changes** (additive only)

- `__init__`: `self.metrics = MetricsCollector(num_nodes=net.num_nodes)`.
- `run()`: after each successful `_run_round()` and `self.t` increment, call
  `self.metrics.record_round(self.net, self.t, self._collect_family_metrics())`.
  After the loop, call `self.metrics.finalize()`.
- New method `_collect_family_metrics(self) -> dict` — default returns `{}`.
- `t_no_dead` is retained; `fnd` should equal it (cross-check, not duplicate).
- **GitNexus impact:** `run()` and `__init__` are inherited by all 12 algorithms.
  Run `gitnexus_impact({target: "run", direction: "upstream"})` (and for
  `__init__`) before editing, and report the blast radius. Changes are additive
  and must not alter existing round behavior.

**3. Family hooks** (override `_collect_family_metrics`)

- Clustering algorithms return
  `{'ch_count': int, 'cluster_sizes': list[int]}`.
- Topology-control algorithms return
  `{'avg_degree': float, 'avg_tx_power': float, 'lambda2': float | None}`.
- Each override reads only already-computed per-round state; it must not mutate
  network state or trigger extra computation that changes energy use.

**4. `benchmark.py` — sweep runner (parallel)**

- Config constants at top of file:
  - `ALGOS` — all 12 algorithm keys (same identifiers as `main.py`'s selector).
  - `DEPLOYMENTS = ['poisson', 'uniform', 'grid', 'gaussian', 'edge']`
  - `SEEDS = [1, 2, 3]` (tunable)
  - `NUM_NODES = 200`
  - `RESULTS_DIR = 'results'`
  - `WORKERS = os.cpu_count()` (tunable; `1` forces sequential for debugging).
- **Task model — embarrassingly parallel.** Each `(algo, deployment, seed)` run
  is fully independent (no shared state; own network, own RNG, own metrics, own
  output file), so runs are distributed across worker **processes**.
- A top-level, picklable worker function `run_one(algo, deployment, seed,
  num_nodes, results_dir) -> path | None`:
  - Returns early (skip) if `results/runs/<algo>_<deploy>_<seed>.json` exists
    (resumability + idempotency, also makes a parallel re-run safe).
  - Sets `MPLBACKEND=Agg` and builds the network via the seeded
    `build_network(deployment, num_nodes, seed)` helper (reuse/extract from
    `main.py`).
  - Instantiates the algorithm with `plot_period = ∞` (no plot windows),
    `run()`s, then writes `results/runs/<algo>_<deploy>_<seed>.json`
    (= `time_series()` + `summary()` + metadata: algo, deployment, seed,
    num_nodes). Each worker writes **only its own file** — no shared-file
    contention.
- The driver builds the full task list (cartesian product), dispatches via
  `concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)`, and reports
  progress as futures complete. **Processes, not threads** — runs are CPU-bound,
  so the GIL would serialize a thread pool.
- **`summary.csv` is built once, after the pool drains**, by reading all
  `results/runs/*.json`. This avoids concurrent writers racing on a shared CSV
  and keeps the summary consistent regardless of completion order.
- **Resumable & safe to re-run:** existing run files are skipped, so a crashed or
  partial sweep can be resumed simply by re-running. Intended to be launched in
  the background (the DIA sweep is slow — ~140k better-response steps to
  converge — but now overlaps with other runs).
- **Determinism is preserved under parallelism:** each run's result depends only
  on its own `(deployment, num_nodes, seed)` seeded RNG, independent of
  scheduling order or worker count.

**5. `plot_benchmark.py` — figures from persisted data** (never re-runs sweep)

- Reads `results/runs/*.json` (+ `summary.csv` for scalars).
- Figures, each one line/bar group per algorithm:
  - Survival curves (alive vs round), per deployment — mean across seeds with
    shaded ±std band.
  - Total residual-energy curves, per deployment.
  - Grouped bar charts: FND / HND / LND.
  - Bar charts: total packets delivered, cumulative PDR, energy-per-packet.
- Saves PNG to `results/figures/`.

### File layout

```
metrics.py                 # MetricsCollector
benchmark.py               # sweep runner -> results/
plot_benchmark.py          # figures from results/
results/
  runs/<algo>_<deploy>_<seed>.json   # per-run time series + summary
  summary.csv                        # one row per run (scalars)
  figures/*.png
```

## Constraints & invariants

- **Shared energy model is invariant** — metrics read state only; no algorithm
  reimplements energy formulas. Delivery uses the existing
  `build_routing_tree()`. (CLAUDE.md energy-model constraint.)
- **Additive changes to `BaseAlgorithm`** — must not change existing round
  semantics or energy consumption; run GitNexus impact analysis first.
- **Reproducibility** — a run is fully determined by
  `(deployment, num_nodes, seed)` via the single seeded RNG in `build_network`.
- **No new heavy dependencies** — numpy/matplotlib already present; CSV/JSON via
  stdlib.

## Out of scope (YAGNI)

- Markdown summary tables (derivable from `summary.csv` later).
- **Multi-machine / distributed** sweep execution. Single-machine multiprocess
  parallelism is in scope (component 4); cross-host orchestration is not.
- Probabilistic channel-loss model (delivery is connectivity-based).
- Protocol-faithful per-algorithm packet accounting (rejected for fairness —
  Approach B).

## Scale estimate

12 algos × 5 deployments × 3 seeds = **180 runs**. Seed count is configurable;
start small and increase once the harness is validated.

## Verification

- `metrics.py` unit-testable in isolation: feed a synthetic `NetworkModel` /
  alive sequence, assert FND/HND/LND and PDR derivations.
- Smoke test: run one algorithm for a few rounds (`max_rounds = 3`,
  `plot_period = ∞`) and confirm a populated `summary()` and JSON file.
- Cross-check `metrics.fnd == algo.t_no_dead`.
