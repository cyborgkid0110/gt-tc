# Simulation-Setup: Scenario Generation + Paper "Simulation Setup" Subsection

**Date:** 2026-06-11
**Scope:** (1) Generate the six evaluation deployment scenarios into `scenarios/gen/`,
(2) render a single 2×3 deployment figure, and (3) write the "Simulation setup"
subsection of the paper's *Results and discussion* section in the LaTeX project at
`~/Documents/workspace/github/GT2_paper/sec4_simulation.tex`.

---

## Problem

The paper's *Results and discussion* section needs a *Simulation setup* subsection
that (a) lists the benchmarked clustering and topology-control algorithms, (b)
describes the deployment scenarios with a figure, and (c) gives the algorithm
parameter tables. The evaluation uses six concrete deployment configurations
grouped into three scenario *types*; results are averaged over 10 random instances
per type. Two of the six configurations already exist as frozen scenarios; the
other four (coverage-deployed, two with a central obstacle) do not, and the
`Region` geometry layer has no support for a circular obstacle.

---

## Scenario inventory

All six configurations live on the existing ±250 m area (500 m × 500 m span) for
comparability. Seeds are `{1,2,3,4,5,6,7,8,9,42}` (10 instances each), matching the
existing batches.

| # | Type | Placement | Nodes | cov_radius | Obstacle | BS | File pattern | Status |
|---|------|-----------|-------|-----------|----------|----|--------------|--------|
| 1 | free space | random uniform | 200 | — | none | (0,0) | `uniform_n200_s*.csv` | **exists** |
| 2 | target region | random gaussian | 100 | — | none | (0,0) | `gaussian_n100_s*.csv` | **exists** |
| 3 | free space | coverage-PSO | 60 | 60 m | none | (0,0) | `cov_free_n60_r60_s*.csv` | new |
| 4 | free space | coverage-PSO | 40 | 90 m | none | (0,0) | `cov_free_n40_r90_s*.csv` | new |
| 5 | obstacle | coverage-PSO | 60 | 60 m | circle r=80 @ (0,0) | (180,−180) | `cov_obs_n60_r60_s*.csv` | new |
| 6 | obstacle | coverage-PSO | 40 | 90 m | circle r=80 @ (0,0) | (180,−180) | `cov_obs_n40_r90_s*.csv` | new |

The paper presents **three scenario types** — free space (1, 3, 4), target region
(2), obstacles (5, 6) — without detailing the generation method. Results are
averaged over the 10 instances of each type's configurations.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Placement for 3–6 | Coverage-maximizing PSO (`coverage_deploy.place_nodes`) | Engineered coverage under a BS-connectivity constraint; the `coverage_radius` is the deployment comm/coverage range |
| Circular obstacle | New `circles: [[cx,cy,r]]` key in the def YAML | Cleaner than hand-writing a 64-point polygon ring; one small `Region.from_def` change |
| Obstacle radius | 80 m at region center | Moderate central hole that forces routing detours |
| BS (scenarios 5,6) | (180, −180) | Bottom-right, inset from the (250,−250) corner |
| BS (scenarios 3,4) | (0,0) | Consistent with free-space scenarios 1,2 |
| Area | ±250 m for all six | Comparability with the existing frozen scenarios |
| Batch generation | New `scenarios/freeze_coverage_scenarios.py`, parallel + idempotent | PSO is heavy (3000 iters × 40 instances); skip existing files |
| Figure | Single 2×3 panel PNG into `GT2_paper/figures/` | Compact; one `\includegraphics` + caption |
| Algorithm roster | 11 algorithms (EE-TCM excluded), 2 groups + GT2 | Matches `benchmark.py` ALGOS |
| Parameter tables | Two: shared sim/energy/radio + per-algorithm hyperparameters | Requested scope |
| Bibliography | **No `\cite{}` commands** — algorithm names in plain text | User will add citations later; `refs.bib` is empty |

---

## Changes

### 1. `scenarios/regions.py` — circle obstacle support

In `Region.from_def`, after building polygon obstacles from `d.get('obstacles', [])`,
also read `d.get('circles', [])` (each entry `[cx, cy, r]`), convert via shapely
`Point(cx, cy).buffer(r)`, and include them in the obstacle union. No other geometry
behavior changes; `obstacles` (polygon rings) keeps working.

### 2. `scenarios/freeze_coverage_scenarios.py` — new batch driver

New module mirroring `freeze_scenarios.py` in structure:

- Define the four coverage configs inline as dicts (name template, num_nodes,
  coverage_radius, obstacle circle if any, bs, area=250).
- For each config × each seed in `{1..9,42}`, build a per-instance def dict
  (injecting the seed and a per-instance `name` matching the file pattern above),
  build the `Region`, call `place_nodes(region, num_nodes, radius, seed,
  'config/coverage.yaml')`, draw `vpre ~ U(2.7, 4.2)` from the seeded RNG, and
  `save_scenario(...)` to `scenarios/gen/<name>.csv`.
- Idempotent: skip a config/seed whose CSV already exists.
- Parallel across the 40 instances via `ProcessPoolExecutor` (picklable top-level
  worker), since each PSO run is independent. Run in the background.
- Factor the single-instance logic into a `freeze_one_coverage(config, seed)`
  helper reused by both the pool and any direct call. `make_coverage_scenario.py`'s
  body can optionally be refactored to call the same helper, but that is not
  required for this spec.

`save_scenario` meta carries `source`, `num_nodes`, `seed`, `area`,
`coverage_radius`; the BS position is written from `region.bs_pos`.

**PSO cost note:** `config/coverage.yaml` uses `pso_iters=3000`. Forty independent
PSO runs are parallelized across cores and run in the background. The current
iteration count is kept (no accuracy change); only the harness parallelizes.

### 3. `plot_deployments.py` — draw circles in `--def` mode

In `plot_from_def`, after drawing polygon `obstacles`, also iterate
`d.get('circles', [])` and add a matplotlib `Circle` patch (same obstacle styling)
for each. This keeps single-scenario rendering correct for the new defs.

### 4. New deployment-figure script (combined 2×3 panel)

A small script (e.g. `plot_eval_scenarios.py`, or a function added to
`plot_deployments.py`) renders **one** 2×3 figure with one representative instance
(seed 1) of each of the six configurations:

- Panels 1,2,3,4 (free space / target region): plain node scatter + BS marker.
- Panels 5,6 (obstacle): node scatter + the central circle obstacle + off-centre BS.
- Each panel titled with its type and `(N, coverage_radius)`.
- Output: `~/Documents/workspace/github/GT2_paper/figures/deployment_scenarios.png`
  (dpi ≈ 130), plus a copy under `docs/figures/` for the repo.

Panels 1,2 load the existing `uniform_n200_s1.csv` / `gaussian_n100_s1.csv`; panels
3–6 load the new `cov_*_s1.csv`. Obstacle geometry for 5,6 is taken from the known
config (center circle r=80), not re-derived.

### 5. `~/Documents/workspace/github/GT2_paper/sec4_simulation.tex` — paper text

Append a `\subsection{Simulation setup}` after the existing section header with:

**(a) Algorithm roster.** GT2 introduced as the proposed method (two-stage hybrid:
mixed-strategy clustering game + pure-strategy power-control game). Then two groups,
each algorithm with a one-line description (algorithm names in plain text — **no
`\cite{}` commands**; the user adds citations later):

- *Clustering protocols:* LEACH, GTFR, FL-LEACH-PSO, SCA-Lévy, FC-CRA.
- *Topology-control games:* DIA, MIA, TCLE, EFTCG-1, EFTCG-2.

**(b) Deployment scenarios.** Prose introducing the three types — free space, target
region, with-obstacles — and stating that each type is evaluated over 10 randomly
generated instances and metrics are averaged. References the 2×3 figure
(`\label{fig:deployment_scenarios}`). No generation-method detail.

**(c) Parameter tables.** Two `table` environments (new `.tex` files under the
project's `table/` dir, `\input` from `sec4_simulation.tex`, or inlined — match the
project's existing `\begin{table}` style with `\hline\hline` rules):

- **Shared simulation parameters** (from `main.py`): area ±250 m; initial energy
  E₀ = 5 mJ; power bounds p_min = 3×10⁻⁵ W, p_max = 2.5×10⁻⁴ W, p_step = 3×10⁻⁶ W;
  hop_max = 3; link-budget radio params — SNR = 10 (10 dB), NF = 6.31 (8 dB),
  N₀ = 3.98×10⁻²¹ W/Hz, BW = 3 MHz, λ = 0.125 m (2.4 GHz), γ = 2.0, G_ant = 1.0
  (0 dBi), η = 0.30, R_bit = 250 kbps; derived P_th ≈ 7.53×10⁻¹³ W; energy model
  E_elec = 50 nJ/bit, E_agg = 5 nJ/bit; packet payloads — data 32 bits, agg 72 bits,
  sensor sample 16 bits; max rounds = 50 000.
- **Per-algorithm hyperparameters** (from `config/*.yaml`):
  - GT2: payoff ρ = 3.4×10⁻⁵, α = 1.5, β = 0.1, μ = 0.01
  - LEACH: p_ch = 0.05
  - GTFR: FCM cluster fraction 0.05, fuzziness m = 2, ψ weights (0.25 each)
  - DIA/MIA: mode (DIA/MIA), dia_k (LDIA, null = global)
  - TCLE: ε = 0.01, pricing = quadratic, μ = 0.01, κ levels = 10, τ = 1.0, σ_max = 0.1
  - EFTCG: k = 1 (EFTCG-1) / k = 2 (EFTCG-2)
  - FL-LEACH-PSO: PSO particles 30, w = 0.7, c1 = c2 = 1.5; gap k_max = 20
  - SCA-Lévy: m = 30, T = 50, a = 2.0, b = 0.5, Lévy β = 1.5, p_ch = 0.05,
    high-energy fraction 0.5
  - FC-CRA: P = 0.05, d_max_factor = 0.5, z_realloc = 0.5

**Bibliography:** Out of scope. Do **not** add `\cite{}` commands or touch
`refs.bib`; algorithm names appear in plain text. The user will add citations later.

---

## Files

| File | Change |
|------|--------|
| `scenarios/regions.py` | Read `circles` key in `from_def`, union into obstacles |
| `scenarios/freeze_coverage_scenarios.py` | **New** — parallel, idempotent batch driver for the 4 coverage configs × 10 seeds |
| `plot_deployments.py` | Draw `circles` in `--def` render mode |
| `plot_eval_scenarios.py` (or fn in `plot_deployments.py`) | **New** — combined 2×3 deployment figure |
| `scenarios/gen/cov_*_s*.csv` | **New** — 40 generated scenario CSVs |
| `~/.../GT2_paper/figures/deployment_scenarios.png` | **New** — figure asset |
| `~/.../GT2_paper/sec4_simulation.tex` | Add `\subsection{Simulation setup}` |
| `~/.../GT2_paper/table/*.tex` (optional) | Shared + per-algo parameter tables |

---

## Verification

- After generation: `ls scenarios/gen/cov_*_s*.csv | wc -l` == 40; spot-check one
  CSV header has the expected `num_nodes`, `coverage_radius`, `bs_x`/`bs_y`.
- Each obstacle scenario: no node falls inside the central circle (distance from
  (0,0) > 80 for all positions) and the BS is at (180,−180).
- Connectivity: `is_connected_to_bs(positions, bs_pos, coverage_radius)` is True for
  every generated CSV (the placer's repair guarantees this).
- Figure: `deployment_scenarios.png` exists and shows six labelled panels with the
  obstacle visible in panels 5,6.
- Paper: `sec4_simulation.tex` compiles within the project (`latexmk`/existing
  build), the figure renders, and the two tables typeset. No new `\cite{}`
  commands are introduced.

---

## Out of scope

- Running the benchmark sweep or producing results plots/tables (this is *setup*
  only — the rest of *Results and discussion*).
- Citations: no `\cite{}` commands and no `refs.bib` edits (user handles later).
- Refactoring `make_coverage_scenario.py` (may share the helper, not required).
