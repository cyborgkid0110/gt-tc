# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Uses Anaconda. Always activate `base` before running Python:
```bash
conda init                       # if not init
conda activate base
```

Required packages: `numpy scipy matplotlib networkx pyyaml` (plus `shapely` for the coverage-deployment generator)

## Running the simulation

```bash
python main.py                   # run GT2 algorithm
MPLBACKEND=Agg python main.py    # run without interactive plot windows
```

Node deployment, count, and seed are CLI flags (see `deployment.py`):
```bash
python main.py --deployment uniform --num-nodes 200 --seed 1 --algo LEACH
# --deployment: poisson (default) | uniform | grid | gaussian | edge
```
A run is reproducible from `(deployment, num_nodes, seed)`.

To test a small number of rounds without plots (useful for smoke-testing changes):
```python
algo = GT2(net, config_path='config/gt2.yaml')
algo.max_rounds = 3
algo.plot_period = 999999
algo.run()
```

## Running the benchmark sweep

Run every (algorithm, deployment, seed) combination in parallel and persist
metrics, then render figures:

```bash
conda run -n base python benchmark.py        # -> results/runs/*.json + results/summary.csv
conda run -n base python plot_benchmark.py   # -> results/figures/*.png
```

Sweep config (algorithms, deployments, seeds, worker count) lives in constants
at the top of `benchmark.py`. Runs are resumable — existing `results/runs/*.json`
are skipped. Metrics are collected by the shared `MetricsCollector` (`metrics.py`)
driven from `BaseAlgorithm.run()`; see
`docs/superpowers/specs/2026-06-01-benchmark-metrics-harness-design.md`.

## Architecture

The codebase separates the **physical model** from the **algorithm**:

- **`model.py`** — `Sensor` and `NetworkModel`. These are algorithm-agnostic and shared by all implementations. `NetworkModel` owns the adjacency matrix, all energy/Friis equations, and topology helpers. `Sensor` holds per-node state. See `docs/model.md` for full API.
- **`deployment.py`** — node-deployment generators (algorithm- and model-agnostic). A `REGISTRY` of pure functions `(num_nodes, area, rng) -> (N, 2)` positions in `[-area, area]²`, one per scenario: `poisson` (default, blue-noise), `uniform`, `grid` (lattice), `gaussian` (clusters), `edge` (radial density toward edges). Scenario knobs are module-level constants with sensible defaults. `generate_positions(name, num_nodes, area, rng)` dispatches; `build_network()` in `main.py` owns the single seeded RNG and threads it through positions **and** `Vpre` for `(deployment, num_nodes, seed)` reproducibility.
- **`algos/__init__.py`** — `BaseAlgorithm` abstract base class. Provides the shared simulation loop (`run()`), round counter (`t`), death tracking (`t_no_dead`, `dead_nodes`, `_track_death()`), and config loading (`max_rounds`, `plot_period`). All algorithms inherit from this and implement `_run_round() -> bool`.
- **`algos/gt2.py`** — `GT2(BaseAlgorithm)`. Two-stage game: mixed-strategy clustering game (CH election) then pure-strategy power-control game (intra-cluster power optimisation toward Nash equilibrium). Reads its hyperparameters from `config/gt2.yaml`.
- **`algos/leach.py`** — `LEACH(BaseAlgorithm)`. Randomised cluster-head rotation with threshold-based election `T(n) = P/(1-P*(r mod 1/P))`, single-hop intra-cluster communication. Reads from `config/leach.yaml`.
- **`algos/gtfr.py`** — `GTFR(BaseAlgorithm)`. Fuzzy C-Means clustering (run once, static) + game-theoretic CH selection via mixed NE (tentative CHs) then fitness function (final CH per cluster). Reads from `config/gtfr.yaml`.
- **`algos/dia_mia.py`** — `DIAMIA(BaseAlgorithm)`. Power-control topology game (no clustering). DIA: restrained better-response (δ-step), O(n²), minmax fair. MIA: greedy best-response, O(n), first-mover bias. LDIA: localized DIA with k-hop reachability (`dia_k` param). Adapts once to convergence, then maintains; re-adapts only on node death. Reads from `config/dia_mia.yaml`.
- **`algos/tcle.py`** — `TCLE(BaseAlgorithm)`. Energy-aware topology control game. Benefit = algebraic connectivity indicator (λ₂ > ε). Cost = integral-based unwillingness with configurable pricing (linear/quadratic/exponential). Block-partitioned strategy set (κ_i ∝ residual energy), wait-time priority for low-energy nodes, event-triggered reconstruction (K unwillingness levels). Reads from `config/tcle.yaml`.
- **`algos/eftcg.py`** — `EFTCG(BaseAlgorithm)`. Energy-efficient and fault-tolerant topology control game. Utility = f_k (k-connectivity indicator) × [α_i × power-saving + β_i × avg-neighbor-energy], with self-adaptive weights α_i = 1 − E_r/E_0. EFTCG-1 (k=1, single connected) prioritises energy; EFTCG-2 (k=2, biconnected) adds fault tolerance. Reads from `config/eftcg.yaml`.
- **`algos/fl_leach_pso.py`** — `FLLEACHPSO(BaseAlgorithm)`. Fuzzy Logic LEACH with PSO. Setup: Gap statistic → optimal k, then hybrid PSO+K-Means clustering (run once). Steady: two-tier CH selection via Mamdani fuzzy logic — PCH (3 inputs, 27 rules) and SCH (2 inputs, 9 rules) — then CM→SCH→PCH→BS data flow. Reads from `config/fl_leach_pso.yaml`.
- **`algos/sca_levy.py`** — `SCALEVY(BaseAlgorithm)`. Centralised CH election via Sine-Cosine Algorithm with sinusoidal step factor + Lévy mutation on below-average individuals. Per round: high-energy candidate pool → `k_opt = round(N_alive · p)` heads → population of `m` 2-D groupings evolved for `T` iterations (continuous positions snapped to candidate sensors). Fitness = intra-cluster distance-variance (Eq. 15). Full multi-hop data path: CM→CH via layered batches, then multi-hop CH→CH→BS by monotone progress toward origin. Paper's relay-node design (Eq. 16) is kept as a reference helper (`_select_relay_for_ch`) but not on the main flow. Reads from `config/sca_levy.yaml`.
- **`algos/fc_cra.py`** — `FCCRA(BaseAlgorithm)`. Adaptive-radius clustering: per-node `R(i) = (1 + β_i)·α_i·R_0` shrinks near the BS (β_i via energy-dispersion `D_E` + near-BS distance factor), shrinks as the network ages (`R_0 ∝ √(E_th / N_alive)`), and compensates local density (`α_i ∝ 1/√N_0(i)`). Iterative CHCC-greedy CH election, persistent clusters rebuilt only when a CH's residual energy falls below `z_realloc_pct` of its snapshot. Multi-hop CH→CH→BS forwarding via `ch_neighbors` (monotone progress to origin). Paper's PEF intracluster routing (§IV-C) and ICCNS Dijkstra intercluster routing (§IV-D) are implemented as reference helpers (`_intra_pef_route`, `_inter_iccns_route`) but not on the main flow. Reads from `config/fc_cra.yaml`.
- **`algos/ee_tcm.py`** — `EETCM(BaseAlgorithm)`. Two-stage protocol. Clustering (§3.4): CH eligibility via `E(s) > β_opt · E_toSink` with `β_opt = ((r_max − t)/r_max) · E_toSink/E_0`, followed by Bernoulli subsample to `p_ch_fraction` and nearest-CH join. Topology-control game (§3.5): per-cluster sequential better-response power reduction accepting moves that improve `u_i = f_k · (α_i · (p_max − p_i)/p_max + β_i · Ē_neighbors)` with self-adaptive `α_i = 1 − E_r/E_0`, `β_i = 1 − α_i` — same utility as EFTCG but scoped per cluster. `f_k` checks strong connectivity on the directed cluster graph (+ biconnectivity if `k=2`). Optional data compression (off by default) reduces CM maintenance by `1/a`. Reads from `config/ee_tcm.yaml`.
- **`main.py`** — entry point. Defines **global parameters** shared across all algorithms (node count, area, energy constants, radio parameters), generates the node population, constructs `NetworkModel`, instantiates the chosen algorithm (controlled by `ALGORITHM` variable), and calls `.run()`.
- **`graph.py`** — NetworkX utilities for BFS layer assignment within clusters. Returns `{ch_pos: [[batch, ...], ...]}` keyed by CH position tuples.
- **`plot.py`** — three visualisation functions (`directional_wsn_plot`, `cluster_head_probability_plot`, `tx_power_plot`) that consume the legacy dict format.
- **`plot_style.py`** — shared paper-figure styling. `apply_paper_style()` sets global rcParams; `PANEL_SIZE` / `WIDE_SIZE` give `figsize`; `colour_map()` assigns a distinct colour per algorithm (11 algos exceed the default 10-colour cycle) keyed off the **sorted** algorithm names so a given algorithm keeps one colour across every figure. Imported by `plot_benchmark.py`, `plot_residual_energy.py`, and `plot_eval_scenarios.py`.

### Paper figure sizing

The paper places result figures inside `0.48\linewidth` subfigures, i.e. ~3.0 in
wide on the page. A figure rendered 8 in wide is downscaled ~0.38×, shrinking a
10 pt label to ~4 pt in print. Figures are therefore rendered close to their
printed size (`PANEL_SIZE = (4.3, 3.1)`) with a larger base font (13 pt), so
text lands at ~9 pt on the page.

Consequences for the tiled 6-panel figures (`survival_*`, `energy_*`,
`energy_before_fnd_*`): at that size a per-panel 11-entry legend covers the
curves and overflows the page, so those panels are generated with no legend and
no title (`_curve(..., legend=False, title=False)`). The LaTeX subfigure
`\caption*` names the scenario, and `plot_benchmark.legend_strip()` writes one
shared `algo_legend.png` that each figure environment includes once. Axis labels
are abbreviated (`$E_\mathrm{res}$ (J)`, `$H_\mathrm{avg}$`) because the full
phrases do not fit a 3.1 in tall panel.

Two paper trees consume these figures and tile them differently, so the
generators take two environment variables:

| Variable | Effect |
|----------|--------|
| `PANEL_COLS` | `2` (default) renders `PANEL_SIZE` for `0.48\linewidth` subfigures, 2 per row. `3` renders `NARROW_PANEL_SIZE` for `0.32\linewidth`, 3 per row, and thins the x ticks (`plot_style.thin_xticks`) so five-digit round labels do not collide. |
| `PAPER_DIR` | Paper tree `plot_residual_energy.py` mirrors into (default `~/…/GT2_paper`). |

`~/…/GT2_paper` uses the 2-per-row layout; `~/…/els-cas-templates` (Elsevier
`cas-sc`, `\textwidth` = 468 pt) uses 3-per-row for `fig:energy` and
`fig:residual`. Regenerate the latter with:

```bash
PANEL_COLS=3 PAPER_DIR=~/Documents/workspace/github/els-cas-templates \
  conda run -n base python plot_benchmark.py
```

### Data-flow through a round

```
net.reset_round()
  → net.discover_neighbors()              # fills adjacency matrix + neighbor lists
  → GT2._clustering_game()               # elects CHs via mixed-strategy NE probability
  → GT2._cluster_formation()             # CMs join nearest CH
  → GT2._filter_neighbours()             # remove cross-cluster edges
  → GT2._connect_unaffiliated()          # isolated nodes raise power to join
  → net.create_cluster_subgraph()        # strip long-range CH links for BFS
  → graph.divide_network_by_clusters()   # layered batch structure per cluster
  → GT2._power_control_game()            # iterative best-response power reduction
  → GT2._maintenance()                   # deduct energy, track deaths
```

### Legacy compatibility

`graph.py` and `plot.py` use the legacy dict format (`node_dict` keyed by `(x, y)` tuples, `network` dict with `vertices` and `edges`). `NetworkModel` provides `to_network_dict()` and `to_node_dict()` to produce these. When `graph.py` returns node positions from the layered-batch structure, convert them back to `Sensor` objects via `net.sensor_by_pos(tuple(float(x) for x in pos))`.

### Connectivity convention: directed edges, strong connectivity validation

`NetworkModel` uses a **directed** adjacency matrix. `discover_neighbors()` creates unidirectional edges: `A→B` exists if `distance(A,B) ≤ A.rc`. The matrix is generally **asymmetric** (A→B does not imply B→A).

**Unidirectional links are valid** — multi-hop directed paths can provide bidirectional reachability. For example, if A→B→C and C→D→E→A both exist, then A and C can communicate bidirectionally via different routes.

**Connectivity checks must use strong connectivity** on the directed graph (every node can reach every other via directed paths), not weak connectivity on an undirected approximation. Specifically:
- **TCLE** (`_compute_algebraic_connectivity`): checks `nx.is_strongly_connected(DiGraph)` before computing λ₂ on the undirected version.
- **EFTCG** (`_check_k_connectivity`): checks `nx.is_strongly_connected(DiGraph)` for k=1; additionally checks `nx.is_biconnected(undirected)` for k=2. The directed graph **includes the base station as a vertex** (`_build_directed_graph`), so `f_k` enforces sink connectivity: forward edge `i→BS` when `dist_to_bs(i) ≤ rc_i`, reverse edge `BS→i` when `i` is within the BS's `p_max` range (BS is recharged infrastructure). Without the BS in the graph, EFTCG would keep node-to-node connectivity while partitioning the network from the sink. See `docs/eftcg.md`.
- **DIA-MIA**: uses directed BFS reachability (`_compute_reachability`) which naturally traverses directed edges — correct by construction.

**When removing outgoing links during adaptation** (node reduces power, loses A→B), do NOT also remove the reverse link B→A. The reverse link may still be valid (B's power is sufficient) and may participate in directed paths that provide connectivity for other nodes.

### Key design constraint: no `deepcopy` on `Sensor`

`Sensor` objects hold mutual `neighbors` references, creating circular graphs. `copy.deepcopy` will recurse infinitely. When saving a local sub-graph snapshot, copy only the edges array:
```python
s.local_net = {'vertices': new_local['vertices'], 'edges': new_local['edges'].copy()}
```

## Parameter separation

| Location | Contains |
|----------|----------|
| `main.py` | Global WSN constants: `NUM_NODES`, `AREA`, `E0`, power limits, wavelength, signal threshold, energy model params (`E_ELEC`, `E_AGG`, packet sizes), simulation params (`MAX_ROUNDS`, `PLOT_PERIOD`). CLI flags `--deployment`/`--num-nodes`/`--seed` override deployment, count, seed. |
| `deployment.py` | Per-scenario node-generation knobs (constants): `POISSON_RADIUS`, `GRID_JITTER`, `GAUSS_BLOBS`, `GAUSS_STD_FRAC`, `EDGE_EXP` |
| `config/gt2.yaml` | GT2-specific: `payoff` (ρ), `alpha`, `beta`, `mu` |
| `config/leach.yaml` | LEACH-specific: `p_ch_fraction` |
| `config/gtfr.yaml` | GTFR-specific: FCM params, psi weights (α,β,γ,δ), fitness lookup tables |
| `config/dia_mia.yaml` | DIA-MIA-specific: `mode` (DIA/MIA), `dia_k` (null or int for LDIA) |
| `config/tcle.yaml` | TCLE-specific: `epsilon`, `pricing`, `mu`, `kappa_levels`, `tau`, `sigma_max` |
| `config/eftcg.yaml` | EFTCG-specific: `k_connectivity` (1 or 2) |
| `config/fl_leach_pso.yaml` | FL-LEACH-PSO-specific: PSO params (`pso_particles`, `pso_w`, `pso_c1`, `pso_c2`), K-Means, Gap statistic |
| `config/sca_levy.yaml` | SCA-Lévy-specific: `population_size` (m), `max_sca_iter` (T), `sca_a`/`sca_b` (r1 bounds), `levy_beta`, `p_ch_fraction`, `high_energy_fraction` |
| `config/fc_cra.yaml` | FC-CRA-specific: `P` (datum CH fraction), `d_max_factor` (near-BS radius = `factor · calc_comm_range(p_max)`), `z_realloc_pct` (rebuild trigger), `chcc_avg_mode` |
| `config/ee_tcm.yaml` | EE-TCM-specific: `p_ch_fraction`, `k_connectivity`, `max_adapt_iter`, `compression_a` (1 = off), `compression_overhead` |

When adding a new algorithm, create `config/<algo>.yaml` for its own parameters and a new class in `algos/<algo>.py`.

## Adding a new benchmark algorithm

1. Create `algos/<name>.py` with a class inheriting from `BaseAlgorithm`.
2. Implement `_run_round(self) -> bool` (return `False` when network is dead).
3. Create `config/<name>.yaml` with algorithm-specific params (`max_rounds` and `plot_period` are global in `main.py`).
4. In `main.py`, import the new class and add it to the `ALGORITHM` selection.

### Energy model constraint

The energy model in `NetworkModel` (`calc_node_cost()`, `calc_tx_cost()`, etc.) is the shared physics layer for all algorithms. **Algorithms must use these methods for energy calculations** — never reimplement or substitute a paper's own energy formulas. The `NetworkModel` assumes bounded transmission distance (`p_min`/`p_max`) and specific energy consumption; this is the common ground for fair benchmarking. Different algorithms may have different clustering or topology-control logic, but the energy model is invariant.

---

## Project Status

### Folder structure

```
gt-tc/
├── algos/               # Algorithm implementations (one class per file)
│   ├── __init__.py      # BaseAlgorithm abstract base class
│   ├── gt2.py           # GT2: two-stage game-theoretic algorithm
│   ├── leach.py         # LEACH: randomised CH rotation protocol
│   ├── gtfr.py          # GTFR: FCM clustering + game-theoretic CH selection
│   ├── dia_mia.py       # DIA/MIA/LDIA: power-control topology game (no clustering)
│   ├── tcle.py          # TCLE: energy-aware topology control with algebraic connectivity
│   ├── eftcg.py         # EFTCG: k-connectivity game with self-adaptive energy weights
│   ├── fl_leach_pso.py  # FL-LEACH-PSO: fuzzy logic + PSO clustering + two-tier CHs
│   ├── sca_levy.py      # SCA-Lévy: Sine-Cosine + Lévy mutation CH election + multi-hop
│   ├── fc_cra.py        # FC-CRA: adaptive-radius clustering + multi-hop CH→BS
│   └── ee_tcm.py        # EE-TCM: β_opt clustering + per-cluster EFTCG-style power game
├── config/              # Per-algorithm YAML config files
│   ├── gt2.yaml
│   ├── leach.yaml
│   ├── gtfr.yaml
│   ├── dia_mia.yaml
│   ├── tcle.yaml
│   ├── eftcg.yaml
│   ├── fl_leach_pso.yaml
│   ├── sca_levy.yaml
│   ├── fc_cra.yaml
│   └── ee_tcm.yaml
├── docs/                # Narrative & API documentation
│   ├── content.md       # Game theory & WSN background + methodology
│   ├── model.md         # Full API reference for model.py
│   ├── gtfr.md          # GTFR algorithm documentation
│   ├── dia_mia.md       # DIA-MIA algorithm documentation
│   ├── tcle.md          # TCLE algorithm documentation
│   ├── sca_levy.md      # SCA-Lévy algorithm documentation
│   ├── fc_cra.md        # FC-CRA algorithm documentation
│   └── ee_tcm.md        # EE-TCM algorithm documentation
├── tests/               # Standalone experiment/test scripts
│   ├── grid_search.py
│   ├── nash.py
│   ├── poisson.py
│   ├── tcgsc.py
│   └── test_numba.py
├── algo.py              # Legacy algorithm (pre-refactor, kept for reference)
├── deployment.py        # Node-deployment generators (poisson/uniform/grid/gaussian/edge)
├── egcr.py              # EGCR benchmark algorithm (not yet ported to new structure)
├── graph.py             # NetworkX cluster/layer utilities
├── leach.py             # Legacy LEACH (pre-refactor, superseded by algos/leach.py)
├── main.py              # Entry point — global params + algorithm runner
├── model.py             # Sensor & NetworkModel (shared physics/energy layer)
├── plot.py              # Visualisation functions
└── script.py            # Legacy runner (pre-refactor)
```

### Project pipeline

The overall goal is to benchmark the GT2 algorithm against standard WSN protocols.

1. **Refactor codebase** ✅ — clean object-oriented model, separate algorithm from physics
2. **Implement GT2** ✅ — two-stage game with clustering + power-control games
3. **Implement benchmark algorithms** — LEACH ✅, GTFR ✅, DIA/MIA/LDIA ✅, TCLE ✅, EFTCG ✅, FL-LEACH-PSO ✅, SCA-Lévy ✅, FC-CRA ✅, EE-TCM ✅, EGCR (not yet ported)
4. **Run experiments & collect metrics** — rounds until first death, total lifetime, energy balance
5. **Produce simulation results** — plots and tables for `docs/content.md` (currently "To be added")

### Tasks completed in this session

| Task | Outcome |
|------|---------|
| Reviewed methodology (`content.md`) and original `main.py` | Identified structure, dead code, and design issues |
| Created `model.py` with `Sensor` and `NetworkModel` classes | Algorithm-agnostic physical/energy layer; replaces global `node_dict` dict pattern |
| Restructured `main.py` | Uses new classes; removed ~80 lines of dead code (`p_strat`, `gamma`, `Vsta`, cost-tracking lists, etc.) |
| Fixed `RecursionError` in power-control game | `Sensor.neighbors` creates circular refs; replaced `copy.deepcopy(local_net)` with shallow dict + `edges.copy()` |
| Created `algos/gt2.py` | `GT2` class with 7 private phase methods and a `run()` entry point |
| Created `config/gt2.yaml` | Extracted GT2-specific params (`payoff`, `alpha`, `beta`, `mu`, `max_rounds`, `plot_period`) |
| Separated global vs algorithm parameters | Global WSN constants stay in `main.py`; algo-specific params go to `config/` |
| Created `docs/model.md` | Detailed explanation of every class, attribute, and method — focused on logic, not syntax |
| Created `CLAUDE.md` | Repo guidance for future Claude instances |
| Created `BaseAlgorithm` in `algos/__init__.py` | Abstract base with shared `run()` loop, death tracking, config loading |
| Refactored `GT2` to inherit from `BaseAlgorithm` | Removed duplicated `run()`, state init, and inlined death tracking |
| Implemented `algos/leach.py` + `config/leach.yaml` | LEACH protocol: threshold CH election, single-hop clusters, shared energy model |
| Implemented `algos/gtfr.py` + `config/gtfr.yaml` | GTFR: FCM clustering (once) + mixed NE TCH selection + fitness-based final CH. 10 balanced clusters for 200 nodes |
| Implemented `algos/dia_mia.py` + `config/dia_mia.yaml` | DIA/MIA/LDIA: power-control topology game. DIA converges in ~140k steps, MIA in 1 pass. LDIA (k-hop) converges 3× faster |
| Implemented `algos/tcle.py` + `config/tcle.yaml` | TCLE: algebraic connectivity (λ₂), unwillingness cost, block-partitioned adaptation, event-triggered reconstruction |
| Updated `main.py` with `ALGORITHM` selector | Supports `'GT2'`, `'LEACH'`, `'GTFR'`, `'DIA'`, `'MIA'`, `'TCLE'`, `'EFTCG-1'`, `'EFTCG-2'`, `'FL-LEACH-PSO'`, `'SCA-LEVY'`, `'FC-CRA'`, `'EE-TCM'` |
| Implemented `algos/eftcg.py` + `config/eftcg.yaml` | EFTCG: k-connectivity game with self-adaptive weights (α_i = 1 − E_r/E_0). EFTCG-1 (single connected) and EFTCG-2 (biconnected). First adaptation stays at p_max (correct NE at full energy); power reduction kicks in after first death. |
| Implemented `algos/fl_leach_pso.py` + `config/fl_leach_pso.yaml` | FL-LEACH-PSO: Gap statistic + hybrid PSO+K-Means (once) + Mamdani fuzzy PCH/SCH selection (per round). Two-tier CH hierarchy (CM→SCH→PCH→BS). Both PCH and SCH use 'CH' energy role with actual tx distances. |
| Implemented `algos/sca_levy.py` + `config/sca_levy.yaml` | SCA-Lévy: per-round CH election via sinusoidal-step SCA + Lévy mutation over a high-energy candidate pool; fitness = intra-cluster distance variance. Full multi-hop maintenance (CM→CH layered batches + CH→CH→BS monotone-progress routing). Eq. 16 relay-node helper kept for reference, not on main flow. |
| Implemented `algos/fc_cra.py` + `config/fc_cra.yaml` | FC-CRA: adaptive per-node cluster radius `R(i)=(1+β_i)·α_i·R_0` (energy-dispersion blend of energy factor + near-BS distance factor), iterative CHCC-greedy CH election, persistent clusters with z-% rebuild trigger, multi-hop CH→CH→BS forwarding. PEF (§IV-C) and ICCNS Dijkstra (§IV-D) implemented as reference helpers only. |
| Implemented `algos/ee_tcm.py` + `config/ee_tcm.yaml` | EE-TCM: β_opt residual-energy CH eligibility + Bernoulli downsample to `p_ch_fraction`; per-cluster sequential better-response power-reduction game with EFTCG-style utility `u_i = f_k · (α_i · power-saving + β_i · avg-neighbour-energy)`; `f_k` = strong connectivity (+ biconnectivity for k=2) on the directed cluster graph. Optional data compression gated by `compression_a > 1` (off by default). |

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **gt-tc** (2388 symbols, 3630 relationships, 81 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/gt-tc/context` | Codebase overview, check index freshness |
| `gitnexus://repo/gt-tc/clusters` | All functional areas |
| `gitnexus://repo/gt-tc/processes` | All execution flows |
| `gitnexus://repo/gt-tc/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
