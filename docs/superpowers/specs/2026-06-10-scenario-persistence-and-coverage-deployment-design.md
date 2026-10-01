# Scenario Persistence & Coverage-Maximising Deployment — Design

**Date:** 2026-06-10
**Status:** Approved (design); implementation pending
**Author:** brainstormed with Claude

## Summary

Two user-facing goals that share one underlying mechanism:

1. **Freeze the existing random scenarios to disk.** Today the five deployment
   scenarios (`poisson`, `uniform`, `grid`, `gaussian`, `edge`) are regenerated
   from `(deployment, num_nodes, seed)` on every run. Persist a generated
   scenario to a CSV file and load it back, so benchmarking reads a fixed file
   instead of re-running the generator.

2. **Add a coverage-maximising custom scenario.** Given a region that may
   contain obstacles, empty spaces, or winding paths (nodes deployable only
   along the paths), place a customisable number of nodes — using a customisable
   *coverage radius* — to maximise area coverage, subject to the deployed
   network staying connected to the base station. The base station may sit
   off-centre but never inside an obstacle.

The unifying insight: **the benchmark should consume frozen scenario files, not
live generators.** Once a scenario is a file on disk, "save the random ones" and
"add a custom one" become the same load path; they differ only in how the file
is produced.

```
                          ┌─ random generators (existing 5) ─┐
  scenario definition ──► │                                  │──► FROZEN CSV ──► build_network() ──► benchmark
                          └─ coverage/PSO generator (new) ────┘     (positions
                                                                     + Vpre + BS)
```

## Decisions (locked during brainstorming)

| Question | Decision |
|----------|----------|
| Base station location | **Make BS a real parameter** of `NetworkModel` (full refactor), default `(0,0)` for backward compatibility. |
| Region representation | **Polygons via shapely** — obstacles as polygons, paths as buffered polylines, deployable area = region minus obstacles. |
| Coverage objective | **Maximise covered area, connectivity to BS required** — every frozen layout must be benchmark-ready. |
| Meaning of the customisable "comm range" | **Coverage radius only (design-time).** Placement uses it for layout; the benchmark's physics comm range (fixed `P_MIN..P_MAX` → ~34–99 m) is unchanged. Connectivity is validated against the real `calc_comm_range(p_max)`. The shared energy model stays invariant. |
| Placement algorithm | **Metaheuristic (projected PSO)**, consistent with the repo's existing PSO (`fl_leach_pso`) and SCA (`sca_levy`) flavour. |
| Spec packaging | **One spec (this file), three implementation plans** (phases below). |

## Decomposition

Three phases. Phases 1 and 2 are independent of each other; phase 3 depends on
both (it needs the CSV format from phase 1 and the off-centre BS from phase 2).

| Phase | Scope | Depends on | Risk |
|-------|-------|-----------|------|
| 1. Scenario persistence | CSV format + `save`/`load` + `freeze` script + load path in `build_network`/`benchmark.py` | — | Low |
| 2. BS-as-parameter | `bs_pos` on `NetworkModel`, `dist_to_bs` helper, replace every `hypot(s.x, s.y)` across `model.py` + ~10 algos | — | **High** (cross-cutting) |
| 3. Coverage generator | shapely region/obstacle/path geometry + projected-PSO placement → writes a CSV with off-centre BS | 1 + 2 | Medium |

---

## Phase 1 — Scenario persistence

### CSV format

One file per scenario, stored under `scenarios/`, named
`<source>_n<num_nodes>_s<seed>.csv` (e.g. `poisson_n200_s42.csv`). Custom
coverage scenarios use their definition name (e.g. `warehouse.csv`).

The node table is pure floats (loadable with `numpy.loadtxt(comments='#')`).
Provenance and the base-station position live in `#` header comments so the data
table stays homogeneous:

```
# gt-tc-scenario v1
# source=poisson num_nodes=200 seed=42 area=250 bs_x=0.0 bs_y=0.0 coverage_radius=
x,y,vpre
12.3,-45.6,3.21
-8.0,17.4,2.95
...
```

- `source` — generator name or coverage-definition name.
- `bs_x`, `bs_y` — base-station position (phase 2 reads this into `bs_pos`).
- `coverage_radius` — set only for coverage scenarios; empty otherwise.
- One node per data row: `x`, `y`, and the per-node `Vpre` (so a frozen scenario
  reproduces the network exactly, including `Vpre`, which currently comes from
  the threaded RNG).

### `scenarios.py` (new module)

```python
def save_scenario(path, positions, vpre, bs_pos, meta): ...
    # positions: (N,2) float; vpre: (N,) float; bs_pos: (2,) float;
    # meta: dict of provenance written as `# k=v` lines.

def load_scenario(path) -> ScenarioData: ...
    # ScenarioData = (positions, vpre, bs_pos, meta)
```

Implemented with the stdlib `csv` module for writing and manual `#`-line parsing
plus `numpy.loadtxt` for reading. No new dependency for this phase.

### Integration

- **`build_network`** gains a scenario-file path. When given one (by name or
  path), it **loads** positions, `Vpre`, and `bs_pos` instead of generating —
  constructs `Sensor`s with the stored `Vpre`, sets `net.bs_pos` (phase 2), and
  still runs `check_potential_connectivity()` as a guard. When not given one, it
  behaves exactly as today.
- **`main.py`** gains `--scenario <name|path>`. A bare name resolves to
  `scenarios/<name>.csv`.
- **`benchmark.py`** prefers a frozen file
  `scenarios/<deployment>_n<N>_s<seed>.csv` when it exists, else falls back to
  live generation. Freezing is therefore opt-in: dropping files into
  `scenarios/` switches the sweep to frozen inputs with no code change.
- **`freeze_scenarios.py`** (new script): for each `(deployment, num_nodes,
  seed)` in a config block mirroring `benchmark.py`'s sweep, build the network
  via the existing generators and write the CSV. Re-running is idempotent
  (skip existing files), matching the benchmark's resumable convention.

---

## Phase 2 — Base station as a parameter

Today the BS is implicitly at the origin: every `math.hypot(s.x, s.y)` means
"distance to BS", in `model.py` (`calc_node_cost` clustering branch,
`check_potential_connectivity`, `build_routing_tree` gateways) and across all
algorithms (`gt2`, `leach`, `gtfr`, `ee_tcm`, `sca_levy`, `fl_leach_pso`,
`fc_cra`, …). This refactor makes the BS position explicit.

### Changes

- `NetworkModel.__init__(..., bs_pos=(0.0, 0.0))`; store `self.bs_x`, `self.bs_y`.
- Add `NetworkModel.dist_to_bs(self, sensor) -> float` returning
  `hypot(sensor.x - self.bs_x, sensor.y - self.bs_y)`.
- Replace every `math.hypot(s.x, s.y)` "distance to BS" call site:
  - In `model.py`, use `self.dist_to_bs(s)`.
  - In each algorithm, use `net.dist_to_bs(s)`.
- Leave genuine node-to-node distances (`Sensor.distance_to`) untouched — only
  the BS-distance sites change.

### Safety & process (per CLAUDE.md / GitNexus rules)

- Run `gitnexus_impact` on `NetworkModel`/`calc_node_cost`/`build_routing_tree`
  and report blast radius before editing.
- Default `bs_pos=(0,0)` keeps all existing scenarios, tests, and the five random
  generators byte-for-byte identical (origin BS).
- After the refactor, re-run `tests/test_benchmark_smoke.py`,
  `tests/test_metrics*.py`, and a one-algo benchmark to confirm no regression.

---

## Phase 3 — Coverage-maximising deployment generator

### Region geometry — `regions.py` (new module, shapely)

A scenario definition under `scenarios/defs/<name>.yaml`:

```yaml
name: warehouse
area: 250                 # square bounds [-area, area]^2 (same scale as others)
num_nodes: 80
coverage_radius: 40       # design-time coverage disk radius (metres)
seed: 7
bs: [-120.0, 30.0]        # base-station position (must lie in deployable area)
obstacles:                # list of polygons (closed rings), [[x,y], ...]
  - [[-50,-50], [50,-50], [50,50], [-50,50]]
paths:                    # optional; when present, nodes deploy ONLY on paths
  - { coords: [[-200,-200], [-100,0], [50,80], [200,200]], width: 30 }
```

`Region` builds:

- `bounds` = the `[-area, area]^2` square (a shapely polygon).
- `obstacles` = union of obstacle polygons.
- `deployable` = `union(buffer(path, width/2) for path in paths) − obstacles`
  when `paths` is non-empty, else `bounds − obstacles`.
- Validates `deployable.contains(Point(bs))` (BS not in an obstacle / on a path
  when paths are used); raises a clear error otherwise.

Helpers: `sample_inside(n, rng)` (rejection-sample feasible points),
`project(point)` (nearest point inside `deployable` via
`shapely.ops.nearest_points`), `coverage_targets(spacing)` (a fixed dense point
set used to score coverage).

### Placement — projected PSO

Config lives in `config/coverage.yaml` (swarm size, iterations, `w`/`c1`/`c2`,
penalty weight `lambda`, target-grid spacing).

- **Decision variables:** the `N` node positions (2·N continuous dims).
- **Warm start:** initialise every particle by `region.sample_inside(N, rng)`
  so all particles begin feasible. (Critical — random init in a winding-path
  region is almost all infeasible.)
- **Fitness:** `coverage_fraction − lambda · disconnected_fraction`, where
  - `coverage_fraction` = fraction of `coverage_targets` within `coverage_radius`
    of some node;
  - connectivity graph has an edge between two nodes (and node↔BS) when their
    distance ≤ `r_conn = net.calc_comm_range(p_max)`; `disconnected_fraction` =
    fraction of nodes not in the BS-rooted connected component.
- **Constraint handling:** after each velocity/position update, `project` every
  node back into `deployable`, so particles never leave the feasible region.
- **Termination:** max iterations or a stall window.
- **Hard connectivity repair (post-process):** if the best solution still has
  disconnected nodes, snap stragglers toward the BS component (along the
  deployable area) until connected. If connectivity is unachievable for the given
  `(N, region, r_conn)`, raise — mirroring `build_network`'s "no connected
  deployment" failure rather than emitting an unusable scenario.
- **Output:** positions + per-node `Vpre` (sampled from the same
  `uniform(2.7, 4.2)` via the definition's seeded RNG) + `bs_pos` → written with
  `save_scenario` (phase 1), `coverage_radius` recorded in the header.

### Driver

`make_coverage_scenario.py --def scenarios/defs/<name>.yaml` → builds the region,
runs PSO, repairs connectivity, and writes `scenarios/<name>.csv`. The benchmark
then consumes it through the same frozen-CSV path as every other scenario.

### Caveat (recorded deliberately)

Optimising 2·N continuous coordinates over a non-convex (path) region is
high-dimensional; quality and runtime scale with swarm × iterations ×
target-grid size. Recommend modest `N` (≈ ≤100) for coverage scenarios, rely on
the feasible warm start, and treat the connectivity repair as the guarantee of
benchmark-readiness rather than the optimiser alone.

---

## Testing

- **Phase 1:** round-trip (`save` then `load` returns identical positions/`Vpre`/
  `bs_pos` within float tolerance); a frozen scenario reproduces the same network
  as live generation for the same `(deployment, N, seed)`; `benchmark.py` picks
  the frozen file when present.
- **Phase 2:** regression — existing smoke/metrics tests pass unchanged with the
  default origin BS; a new test with `bs_pos ≠ origin` confirms distances and
  energy costs use it; one-algo benchmark smoke passes.
- **Phase 3:** region tests (deployable excludes obstacles; paths constrain
  placement; off-obstacle BS validated, in-obstacle BS rejected); PSO on a toy
  region yields a connected, in-region layout above a coverage threshold;
  determinism for a fixed seed.

## Dependencies

- Add **`shapely`** to required packages (phase 3 only). Phases 1–2 add no new
  dependency.

## Out of scope

- Varying the benchmark's physics comm range per scenario (explicitly rejected —
  the energy model stays invariant).
- A GUI/interactive region editor; regions are authored as YAML definitions.
- Re-running or re-tuning the existing algorithms for off-centre BS performance;
  phase 2 only makes the BS configurable and correct, not optimised.
