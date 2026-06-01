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
| **Average tx power / radius** | mean transmission radius `rc` over alive nodes |
| **Algebraic connectivity λ₂ / k-connectivity** | structural robustness (TCLE/EFTCG already compute these) |

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
  - `delivered` = `len(net.build_routing_tree())`
  - `generated` = `alive`
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

**4. `benchmark.py` — sweep runner**

- Config constants at top of file:
  - `ALGOS` — all 12 algorithm keys (same identifiers as `main.py`'s selector).
  - `DEPLOYMENTS = ['poisson', 'uniform', 'grid', 'gaussian', 'edge']`
  - `SEEDS = [1, 2, 3]` (tunable)
  - `NUM_NODES = 200`
  - `RESULTS_DIR = 'results'`
- For each `(algo, deployment, seed)`:
  - Build the network via the seeded `build_network(deployment, num_nodes, seed)`
    helper (reuse/extract from `main.py`).
  - Instantiate the algorithm with `plot_period = ∞` (no plot windows;
    `MPLBACKEND=Agg` respected).
  - `run()`, then read `algo.metrics`.
  - Write `results/runs/<algo>_<deploy>_<seed>.json` (= `time_series()` +
    `summary()` + metadata: algo, deployment, seed, num_nodes).
  - Append a row to `results/summary.csv` (summary scalars + algo/deploy/seed).
- **Resumable:** skip a run if its JSON already exists. `summary.csv` is rebuilt
  from the per-run JSON files at the end (so re-runs don't duplicate rows).
- Sequential execution; intended to be launched in the background (the DIA sweep
  is slow — ~140k better-response steps to converge).

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
- Parallel/distributed sweep execution.
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
