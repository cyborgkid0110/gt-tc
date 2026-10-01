# Benchmark Results Analysis & Paper Write-up — Design

**Date:** 2026-06-13
**Status:** Approved (pending spec review)
**Goal:** Turn the completed benchmark sweep into the *Results and Discussion*
section (§5.2–5.4) of `GT2_paper/sec4_simulation.tex`, organised around three
metric criteria and two analytical lenses, and answer the EFTCG
energy-per-packet anomaly.

---

## 1. Background & motivation

The benchmark sweep (`conda run -n base python benchmark.py`) has produced
`results/runs/*.json` (660 runs: 11 algorithms × 6 scenarios × 10 seeds),
`results/summary.csv`, `results/summary_by_scenario.csv`, and figures under
`results/figures/`.

The paper's Results section (`sec4_simulation.tex`, rendered as **§5**) has the
setup written (§5.1) and three empty subsections to fill:

- **§5.2 Network lifetime**
- **§5.3 Network behaviour**
- **§5.4 Network performance**

Two findings from exploration shape this work:

1. **EFTCG energy anomaly is real but scenario-specific and outlier-driven.**
   EFTCG-1/EFTCG-2 have the *lowest* energy-per-packet in 5 of 6 scenarios. The
   "significantly higher" appearance comes only from the **aggregate**
   `energy_per_packet.png`, which is dominated by `gaussian_n100`. Within that
   scenario the mean is inflated to ~0.1 J/packet by **2 of 10 seeds (3 & 7)**
   where total delivered = 100 packets.

   **Root cause (verified from the per-round trace):** at round 0 (initial
   loaded power) delivery is 100/100 — consistent with the
   `check_initial_connectivity.py` guarantee. EFTCG's **first topology-control
   adaptation** (round 1) reduces power, and delivery collapses to 0 for the
   remaining ~36 000 rounds while nodes stay alive draining energy. EFTCG's
   utility `f_k · [α·power-saving + β·neighbour-energy]` checks **node-to-node**
   k-connectivity; **the base station is not a vertex in that graph.** In a
   clustered (gaussian) layout the BS is reached only through the few long links
   pointing toward the centre — exactly the links the power-saving term trims
   first — so the sensor graph stays connected but the **path to the sink is
   severed**. Clustering protocols never hit this because the BS is the root of
   their data-collection tree by construction. This is one item in a broader
   per-algorithm analysis, **not** the centrepiece.

2. **Two requested metrics are not in the collected data and cannot be
   reconstructed from existing JSONs.** Average hop-to-BS over time is not
   recorded; average transmit power is recorded for topology algorithms only
   (not clustering). Both require a `metrics.py` extension and a re-run.

---

## 2. Scope

Three parts, executed in order. The terminal deliverable is **the figures and
tables, plus a bullet-point writing guideline for §5.2–5.4**. The user writes
the full LaTeX analysis prose themselves; this work hands them the data, the
visuals, and a structured outline of what to write in each subsection.

### Part A — Metric harness extension + re-run

Extend `metrics.py` so every algorithm (clustering and topology) reports the two
missing metrics, then re-run the sweep.

**Universal-core additions** (computed in `MetricsCollector.record_round`, so
every algorithm gets them regardless of family):

- **`avg_hop`** — mean hop-depth to the BS over delivered nodes. The routing
  tree is already built each round for `delivered`; reuse it:
  `tree = net.build_routing_tree()` returns per-node `depth`. Record
  `avg_hop = mean(depth for all nodes in tree)` (gateways depth = 1 to BS).
  Add `avg_hop` to the per-round series and expose it in `time_series()`.
  Refactor `record_round` to call `build_routing_tree()` **once** and derive
  both `delivered` (len) and `avg_hop` (mean depth) from the same tree.
- **`avg_tx_power`** — promote to universal-core. Compute
  `mean(s.power for alive s)` inside `record_round`, add to the per-round series
  and `time_series()`. Leave the existing `topology_family_metrics.avg_tx_power`
  in place for backward compatibility, or remove it to avoid duplication
  (implementation decides; the universal-core one is canonical).

**Family extras unchanged:** `ch_count`/`cluster_sizes` (clustering),
`avg_degree`/`lambda2` (topology) remain as-is.

**Summary additions:** add scalar summaries derived from the new series where
useful for tables — e.g. `mean_avg_hop` (time-averaged hop count),
`mean_avg_tx_power` (time-averaged transmit power). Add corresponding columns to
`finalize()`'s summary dict so they flow into `summary.csv` and
`summary_by_scenario.csv`.

**Re-run procedure:**

1. Back up the current results: move `results/runs/` → `results/runs_backup/`
   (and `summary*.csv` → `*.bak`). The schema changed, so existing JSONs lack
   the new fields and must be regenerated; the backup preserves the old run.
2. `conda run -n base python benchmark.py` (resumable; writes fresh
   `results/runs/*.json`).
3. `conda run -n base python plot_benchmark.py` (regenerates figures + both
   summary CSVs).

**Verification:** spot-check one regenerated JSON has `avg_hop` and
`avg_tx_power` in `time_series` (core, not under `family`); confirm clustering
algorithms now report `avg_tx_power`; confirm `summary_by_scenario.csv` matches
the pre-existing lifetime/energy numbers for an unchanged metric (sanity that
the re-run is consistent).

### Part B — Figures & tables (all per-scenario)

Extend `plot_benchmark.py`. New plots keep deployments **separate** (never
averaged into one aggregate), so normal vs extreme behaviour is visible.

Add a `_family_curve(runs, deployment, family_key, ...)` helper (mirrors
`_curve` but reads `r['time_series']['family'][family_key]`, skipping runs that
lack the key). `avg_hop` and `avg_tx_power` are core-series, so plain `_curve`
works for them.

| Criterion | Output | Source | New? |
|---|---|---|---|
| **Lifetime** | FND/HND/LND grouped bars by scenario | `*_by_scenario.png` | exists |
| | Alive-node survival curves, one fig per scenario | `survival_<dep>.png` | exists |
| **Behaviour** | `avg_hop` over time, one fig per scenario | core `avg_hop` via `_curve` | **new** |
| | CH-count over time, one fig per scenario (clustering algos) | family `ch_count` via `_family_curve` | **new** |
| **Performance** | `avg_tx_power` per node, by-scenario bars + per-scenario over-time curve | core `avg_tx_power` | **new** |
| | Throughput (packets to BS) by scenario | `total_delivered_by_scenario.png` | exists |
| | Energy-per-packet **by scenario** (primary) | `energy_per_packet_by_scenario.png` | exists; **promote** |

**Energy-per-packet handling:** make the **by-scenario** chart the primary
figure in the paper; demote (do not delete) the aggregate `energy_per_packet.png`.
In `summary_by_scenario.csv` add a **median** energy-per-packet column alongside
the mean, so the gaussian outlier's distortion is explicit in the table
(`energy_per_packet_median`). The discussion explains the mean/median gap via the
sink-disconnection failure mode.

**Naming:** new figures follow the existing convention —
`hop_<deployment>.png`, `ch_count_<deployment>.png`,
`tx_power_<deployment>.png`, plus `tx_power_by_scenario.png`.

### Part C — Writing guideline for `sec4_simulation.tex` (§5.2–5.4)

**The user writes the full prose.** This part delivers a **bullet-point
guideline** per subsection: what points to make, which figure/table backs each,
and the concrete numbers/comparisons to cite. Delivered as a markdown file
(e.g. `docs/paper_section5_guideline.md`), not LaTeX. Each subsection is framed
through both lenses:

- **Aspect 1 — GT2 vs baselines:** where and why GT2 wins.
- **Aspect 2 — normal vs extreme deployment:** `uniform_n200` (dense, even) is
  the *normal* case; `gaussian_n100` (dense, uneven/clustered) and the coverage
  scenarios `cov_*_n40/n60` (sparse, with/without central obstacle) are the
  *extreme* cases. The lens emphasises how each algorithm holds up moving from
  normal to extreme.

For each subsection the guideline provides:

- The **claim/point** to write (one bullet each).
- The **figure or table** that supports it (by filename).
- The **specific numbers** to quote (pulled from `summary_by_scenario.csv`),
  e.g. "GT2 FND ≈ 13049 on uniform_n200 vs LEACH 7519 (1.7×)".
- Which **lens** the point serves (GT2-vs-baseline or normal-vs-extreme).

**§5.2 Network lifetime** — bullets covering FND (most important), HND, LND,
survival curves: GT2 vs baselines on time-to-first-death and graceful
degradation; how the ranking shifts from normal to extreme deployments.

**§5.3 Network behaviour** — bullets covering average hop-to-BS over time and
CH-count over time: how GT2's two-stage game (clustering + intra-cluster power
control) shapes hop count and cluster-head population versus baselines; how these
evolve as nodes die; normal vs extreme.

**§5.4 Network performance** — bullets covering average transmit power,
throughput (packets to BS), and energy-per-packet by scenario: per-algorithm
analysis including the **EFTCG sink-disconnection failure** as one observation
among several (sink-agnostic topology game vs GT2's BS-rooted clustering);
GT2 vs baselines; normal vs extreme.

The guideline also notes which figures to copy into the paper's `figures/` dir
and which tables to add under `table/` (following the existing
`\input{table/...}` pattern), so the user can wire them in while writing.

---

## 3. Out of scope

- Re-running with different algorithm hyperparameters or new scenarios.
- New algorithms or changes to the energy/physics model (`NetworkModel`).
- Changing the connectivity/delivery semantics in `metrics.py` beyond adding the
  two new series.
- Writing §5.1 (already complete) or other paper sections.

---

## 4. Risks & mitigations

- **Re-run cost / interruption.** The sweep is resumable (skips existing
  `results/runs/*.json`); a crash mid-run resumes cleanly. Backup preserves the
  prior run regardless.
- **`build_routing_tree` depth semantics.** Confirm the returned `depth` is
  hop-count to BS (model.py:199–204 documents `depth: hop count to BS`). The
  `avg_hop` definition must match what the paper claims ("shortest path to BS").
- **`avg_tx_power` units.** Transmit power is in the model's internal units
  (small, ~1e-4); label axes/tables consistently with the energy-model table in
  §5.1.
- **EFTCG framing.** Keep the failure as one analytical point; do not let it
  dominate §5.4. The headline for EFTCG is "best energy-per-packet in 5/6
  scenarios; fails on sink-disconnection in clustered layouts."

---

## 5. Acceptance criteria

1. `metrics.py` records `avg_hop` and universal `avg_tx_power` per round; new
   scalar summaries flow into both CSVs.
2. Sweep re-run; every JSON (clustering and topology) carries the two new core
   series; prior results backed up.
3. New per-scenario figures generated: hop-over-time, CH-count-over-time,
   tx-power (by-scenario bars + over-time), plus promoted energy-per-packet
   by-scenario; `summary_by_scenario.csv` gains a median energy-per-packet
   column.
4. Bullet-point writing guideline for §5.2, §5.3, §5.4 delivered as a markdown
   file: each point names its supporting figure/table and the specific numbers
   to cite, structured by the two lenses, with the EFTCG finding included as one
   per-algorithm observation. The user writes the LaTeX prose themselves.
