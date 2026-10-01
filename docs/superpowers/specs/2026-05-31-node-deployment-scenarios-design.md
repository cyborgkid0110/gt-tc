# Node Deployment Scenarios — Design

**Date:** 2026-05-31
**Status:** Approved (design phase)

## Problem

Benchmarking currently uses a single node-deployment generator — Poisson-disk
sampling — hardcoded inside `build_network()` in `main.py`. To evaluate the WSN
algorithms across diverse spatial layouts, we need additional deployment
scenarios. Generation must be cleanly decoupled from both the physics layer
(`model.py`) and the runner (`main.py`), and each scenario must be independently
testable.

## Goals

- Support five deployment scenarios: Poisson-disk (existing), uniform random,
  grid/lattice, Gaussian clusters, edge/corner-biased.
- Make node count a runtime parameter (area fixed).
- Select scenario + parameters from the CLI, mirroring the existing `--algo`
  pattern.
- Keep each generator a small, isolated, testable unit.

## Non-Goals

- No change to the energy/physics model, algorithms, or base-station placement.
- No per-scenario YAML config (CLI-only, by decision).
- No exhaustive per-knob CLI flags (sensible hardcoded defaults instead).

## Approach

**Function registry** (chosen over a class hierarchy or inline `if/else`).
Generation is stateless, so pure functions plus a name→function registry are the
lightest modular design. Adding a scenario is one function plus one registry
line. This matches the spirit of the existing `model` / `algo` separation.

## Architecture

### New module: `deployment.py`

Each scenario is a pure function with a uniform signature, returning an
`(N, 2)` float `np.ndarray` of coordinates in `[-area, area]²`:

```python
def poisson_disk(num_nodes, area, rng) -> np.ndarray
def uniform(num_nodes, area, rng) -> np.ndarray
def grid(num_nodes, area, rng) -> np.ndarray
def gaussian_clusters(num_nodes, area, rng) -> np.ndarray
def edge_biased(num_nodes, area, rng) -> np.ndarray

REGISTRY = {
    'poisson':  poisson_disk,   # default — current behavior
    'uniform':  uniform,
    'grid':     grid,
    'gaussian': gaussian_clusters,
    'edge':     edge_biased,
}

def generate_positions(name, num_nodes, area, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return REGISTRY[name](num_nodes, area, rng)
```

**Contract for every generator:**
- Returns exactly `num_nodes` positions.
- All coordinates lie within `[-area, area]` on each axis.
- Deterministic given the same `rng` (hence same `seed`).
- Takes only `(num_nodes, area, rng)`; scenario-specific knobs are module-level
  constants with sensible defaults.

### Scenario defaults (module-level constants in `deployment.py`)

| Scenario | Behavior | Constants / defaults |
|----------|----------|----------------------|
| `poisson` | `qmc.PoissonDisk(d=2, radius=30)` over `[0, 2·area]`, shifted to `[-area, area]`. Shortfall (nodes the disk sampler can't place) topped up with uniform-random so it always returns `num_nodes`. | `POISSON_RADIUS = 30` |
| `uniform` | i.i.d. `rng.uniform(-area, area, (N, 2))`. | — |
| `grid` | `ceil(√N)` points per side evenly spanning `[-area, area]`, take first `N`; optional jitter added per point. | `GRID_JITTER = 0.0` (pure lattice) |
| `gaussian` | `GAUSS_BLOBS` centers drawn uniformly in bounds; each node assigned to a blob and sampled from an isotropic Gaussian; samples clipped to bounds. | `GAUSS_BLOBS = 4`, `GAUSS_STD = area / 6` |
| `edge` | Radial density increasing with distance from origin (energy-hole stressor). Radius sampled as `r = area · u^(1/(EDGE_EXP+1))` over the bounding region, angle uniform; rejection-clipped to the square. | `EDGE_EXP = 2.0` |

These constants are edited in code when a non-default knob is needed
(per the "sensible defaults, few flags" decision).

### `main.py` changes

- `build_network(deployment, num_nodes, seed)`:
  - Calls `generate_positions(deployment, num_nodes, AREA, seed)`.
  - Wraps each position into a `Sensor` (the `e0` / `power` / `Vpre` setup and
    `NetworkModel` construction are unchanged and stay here).
  - `Vpre` is drawn from the **same** `np.random.default_rng(seed)` used for
    positions, so a run is fully reproducible from `(deployment, num_nodes, seed)`.
    To keep position generation pure, `generate_positions` will be refactored to
    accept/return the `rng` (or `build_network` constructs the `rng` once and
    passes it both to the generator and the `Vpre` loop). Implementation detail
    resolved during planning; the observable contract is single-seed
    reproducibility.
- New argparse flags:
  - `--deployment` — `choices = list(REGISTRY)`, default `'poisson'`.
  - `--num-nodes` — `int`, default `200` (the former module-level `NUM_NODES`).
  - `--seed` — `int`, default `42`.
  - Existing `--algo` is untouched.
- Base station remains at the origin `(0, 0)`; coordinates remain in
  `[-area, area]²`. No algorithm sees any change to sink-distance assumptions.

### Tests: `tests/test_deployment.py`

For each generator:
- Returns exactly `num_nodes` positions.
- All coordinates within `[-area, area]` on each axis.
- Determinism: same seed → identical array.
- One distribution-specific sanity check:
  - `poisson` — min pairwise distance ≳ `POISSON_RADIUS` (for non-topped-up nodes).
  - `grid` — regular spacing along each axis.
  - `gaussian` — sample mass concentrated near blob centers.
  - `edge` — node density higher in the outer ring than the inner ring.

Plus a registry test: every `REGISTRY` key resolves to a callable satisfying the
contract.

## Data Flow

```
CLI: --deployment <name> --num-nodes <N> --seed <S>
  → build_network(name, N, S)
      → generate_positions(name, N, AREA, S)   # deployment.py
          → REGISTRY[name](N, AREA, rng)        # (N,2) coords in [-AREA, AREA]
      → wrap positions + Vpre into Sensor[]      # main.py (unchanged logic)
      → NetworkModel(sensors, AREA, ...)         # unchanged
  → make_algo(...).run()
```

## Reproducibility & Compatibility Note

A run is identified by `(deployment, num_nodes, seed)`. Threading a single
`np.random.default_rng(seed)` through both positions and `Vpre` makes node sets
exactly reproducible, but produces numbers that differ from the legacy
`random.seed(42)` path. This is a **one-time baseline shift** for
`--deployment poisson`: the spatial layout from `qmc.PoissonDisk` is unchanged
(it already used a `default_rng`), but `Vpre` values and the uniform-random
top-up for unplaced nodes will differ. Existing benchmark numbers should be
regenerated after this change. (Decision: accept the shift for a clean unified
RNG; legacy bit-for-bit preservation was offered and not required.)

## Risks

- **Edge-biased rejection sampling efficiency** — if rejection rate is high for
  the square clipping, fall back to per-axis transform sampling. Low risk at the
  default `EDGE_EXP`.
- **Grid node count** — `ceil(√N)²` ≥ N; taking the first `N` yields a slightly
  non-rectangular boundary row. Acceptable for a reference layout; documented.
```
