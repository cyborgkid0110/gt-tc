# Scenario Persistence & Coverage Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the benchmark consume frozen scenario CSV files, make the base station a real parameter of the model, and add a coverage-maximising deployment generator for regions with obstacles/paths.

**Architecture:** Three independent-ish phases. Phase 1 adds a CSV scenario format + load path. Phase 2 lifts the hard-coded origin base station into a `NetworkModel.bs_pos` parameter. Phase 3 adds shapely region geometry + a projected-PSO placement that writes a frozen CSV. Phases 1 and 2 are independent; phase 3 depends on both.

**Tech Stack:** Python, numpy, scipy, networkx, pyyaml, shapely (new, phase 3 only). Tests follow the repo convention: plain functions with `assert` + an `if __name__ == '__main__'` runner, run with `conda run -n base python tests/test_*.py`.

**Spec:** `docs/superpowers/specs/2026-06-10-scenario-persistence-and-coverage-deployment-design.md`

---

## Execution constraint (user instruction)

**Do NOT run `git commit` during execution.** Every task ends with a *checkpoint* step that runs the task's tests and confirms green; staging/committing is left entirely to the user. Where this plan would normally say "commit", it says "checkpoint" instead.

Always activate the conda env first: `conda activate base` (or prefix commands with `conda run -n base`).

---

## File Structure

**Phase 1 — persistence**
- Create: `scenarios.py` — `save_scenario` / `load_scenario` + the `ScenarioData` shape. One responsibility: serialise/deserialise a deployment to/from CSV.
- Create: `freeze_scenarios.py` — script that snapshots the five random generators across the benchmark seed grid into `scenarios/*.csv`.
- Modify: `main.py` — `build_network` gains a `scenario` load path; add `--scenario` CLI flag.
- Modify: `benchmark.py` — prefer a frozen CSV when present.
- Create: `tests/test_scenarios.py`.
- Create (dir): `scenarios/` (frozen CSVs land here; gitignored or committed at user's discretion).

**Phase 2 — base station parameter**
- Modify: `model.py` — `NetworkModel.bs_pos` + `dist_to_bs()`; replace origin-distance sites; extend `check_potential_connectivity` to require BS reachability.
- Modify: `algos/*.py` — replace every `math.hypot(node.x, node.y)` (distance-to-BS) with `net.dist_to_bs(node)`.
- Modify: `main.py` — thread `bs_pos` into `NetworkModel` (from scenario or default origin).
- Create: `tests/test_bs_param.py`.

**Phase 3 — coverage generator**
- Create: `regions.py` — shapely `Region` (bounds, obstacles, paths, deployable area, BS validation, sampling/projection/coverage helpers).
- Create: `coverage_deploy.py` — projected-PSO placement + connectivity repair.
- Create: `config/coverage.yaml` — PSO knobs.
- Create: `make_coverage_scenario.py` — driver: region def YAML → frozen CSV.
- Create (dir): `scenarios/defs/` — region definition YAMLs.
- Create: `tests/test_regions.py`, `tests/test_coverage_deploy.py`.

---

# PHASE 1 — Scenario persistence

### Task 1.1: `scenarios.py` round-trip (save/load)

**Files:**
- Create: `scenarios.py`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenarios.py
"""Round-trip and parsing tests for scenarios.py (CSV scenario persistence)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import save_scenario, load_scenario


def test_round_trip():
    """save then load returns identical positions, vpre, bs, and meta."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(-250, 250, size=(20, 2))
    vpre = rng.uniform(2.7, 4.2, size=20)
    bs_pos = (12.5, -33.0)
    meta = {'source': 'poisson', 'num_nodes': 20, 'seed': 42, 'area': 250,
            'coverage_radius': ''}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 's.csv')
        save_scenario(path, positions, vpre, bs_pos, meta)
        pos2, vpre2, bs2, meta2 = load_scenario(path)

    assert np.allclose(positions, pos2), "positions changed on round-trip"
    assert np.allclose(vpre, vpre2), "vpre changed on round-trip"
    assert np.allclose(bs_pos, bs2), "bs_pos changed on round-trip"
    assert meta2['source'] == 'poisson'
    assert int(meta2['num_nodes']) == 20
    assert int(meta2['seed']) == 42
    print("  round-trip OK")


if __name__ == '__main__':
    test_round_trip()
    print("\nAll scenarios tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n base python tests/test_scenarios.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'scenarios'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scenarios.py
"""Persist and load a deployment scenario as a CSV file.

Format (one file per scenario):

    # gt-tc-scenario v1
    # source=poisson num_nodes=200 seed=42 area=250 bs_x=0.0 bs_y=0.0 coverage_radius=
    x,y,vpre
    12.3,-45.6,3.21
    ...

The node table is pure floats (loadable with numpy.loadtxt). Provenance and the
base-station position live in `#` header comments so the table stays homogeneous.
"""
import csv

import numpy as np

SCHEMA_VERSION = "v1"
_META_KEYS_INT = {'num_nodes', 'seed'}


def save_scenario(path, positions, vpre, bs_pos, meta):
    """Write positions (N,2), vpre (N,), bs_pos (2,) and a meta dict to `path`."""
    positions = np.asarray(positions, dtype=float)
    vpre = np.asarray(vpre, dtype=float)
    assert positions.shape[0] == vpre.shape[0], "positions/vpre length mismatch"

    full_meta = dict(meta)
    full_meta['bs_x'] = float(bs_pos[0])
    full_meta['bs_y'] = float(bs_pos[1])
    meta_str = " ".join(f"{k}={full_meta[k]}" for k in full_meta)

    with open(path, 'w', newline='') as f:
        f.write(f"# gt-tc-scenario {SCHEMA_VERSION}\n")
        f.write(f"# {meta_str}\n")
        w = csv.writer(f)
        w.writerow(['x', 'y', 'vpre'])
        for (x, y), v in zip(positions, vpre):
            w.writerow([repr(float(x)), repr(float(y)), repr(float(v))])


def load_scenario(path):
    """Return (positions (N,2), vpre (N,), bs_pos (2,), meta dict)."""
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            body = line[1:].strip()
            if body.startswith('gt-tc-scenario'):
                continue
            for tok in body.split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v

    # `comments='#'` drops the two header comment lines; `names=True` consumes
    # the `x,y,vpre` column-header row and exposes columns by name.
    table = np.genfromtxt(path, delimiter=',', comments='#', names=True)
    positions = np.column_stack([table['x'], table['y']]).astype(float)
    vpre = np.asarray(table['vpre'], dtype=float)
    bs_pos = (float(meta.get('bs_x', 0.0)), float(meta.get('bs_y', 0.0)))
    return positions, vpre, bs_pos, meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n base python tests/test_scenarios.py`
Expected: PASS — `round-trip OK`.

- [ ] **Step 5: Checkpoint (no commit)**

Run the test again, confirm green. Do **not** `git commit`.

---

### Task 1.2: `build_network` learns to load a scenario

**Files:**
- Modify: `main.py` (`build_network`, CLI)
- Test: `tests/test_scenarios.py` (add a case)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scenarios.py`:

```python
def test_build_network_from_scenario():
    """A frozen scenario reproduces a NetworkModel with the stored state."""
    import tempfile
    from main import build_network

    net = build_network('uniform', num_nodes=30, seed=3)
    positions = np.array([[s.x, s.y] for s in net.sensors])
    vpre = np.array([s.Vpre for s in net.sensors])

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'u.csv')
        save_scenario(path, positions, vpre, (0.0, 0.0),
                      {'source': 'uniform', 'num_nodes': 30, 'seed': 3,
                       'area': 250, 'coverage_radius': ''})
        net2 = build_network(scenario=path)

    pos2 = np.array([[s.x, s.y] for s in net2.sensors])
    vpre2 = np.array([s.Vpre for s in net2.sensors])
    assert np.allclose(positions, pos2), "loaded positions differ"
    assert np.allclose(vpre, vpre2), "loaded vpre differ"
    print("  build_network(scenario=...) OK")
```

And call it from `__main__`:

```python
    test_round_trip()
    test_build_network_from_scenario()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n base python tests/test_scenarios.py`
Expected: FAIL — `build_network()` has no `scenario` parameter (`TypeError`).

- [ ] **Step 3: Write minimal implementation**

In `main.py`, add the import near the top:

```python
from scenarios import load_scenario
```

Replace the `build_network` signature and add a load branch at the very start of its body (before the `rng = ...` line):

```python
def build_network(deployment='poisson', num_nodes=NUM_NODES, seed=42,
                  max_resamples=MAX_RESAMPLES, scenario=None):
    """Build a NetworkModel for the given deployment scenario.

    If ``scenario`` (a path to a frozen CSV) is given, positions, per-node Vpre,
    and the base-station position are loaded from disk and generation is skipped.
    Otherwise the network is generated from (deployment, num_nodes, seed) exactly
    as before. ...
    """
    if scenario is not None:
        positions, vpre, bs_pos, meta = load_scenario(scenario)
        sensors = []
        for i in range(len(positions)):
            sensors.append(Sensor(id=i, x=float(positions[i, 0]),
                                  y=float(positions[i, 1]), e0=E0,
                                  power=P_MAX / 4, Vpre=float(vpre[i])))
        net = NetworkModel(sensors, AREA,
                           snr=SNR, nf_rx=NF_RX, n0=N0, bw=BW,
                           wave=WAVE, gamma=GAMMA, g_ant=G_ANT, eta=ETA, r_bit=R_BIT,
                           p_min=P_MIN, p_max=P_MAX, p_step=P_STEP,
                           hop_max=HOP_MAX, e_elec=E_ELEC, e_agg=E_AGG,
                           data_payload=DATA_PAYLOAD, agg_payload=AGG_PAYLOAD,
                           sensor_sample_bits=SENSOR_SAMPLE_BITS,
                           bs_pos=bs_pos)
        if not net.check_potential_connectivity():
            raise RuntimeError(f"Frozen scenario {scenario} is not connectable.")
        return net

    rng = np.random.default_rng(seed)
    # ... unchanged body ...
```

> Note: `bs_pos=bs_pos` requires Phase 2. If Phase 2 is not yet merged, drop the
> `bs_pos=bs_pos` kwarg here (origin BS only) and add it back in Phase 2 Task 2.4.

Add the CLI flag in `__main__` (after the existing `--seed` arg):

```python
    parser.add_argument('--scenario', type=str, default=None,
                        help='Path to a frozen scenario CSV (overrides '
                             '--deployment/--num-nodes/--seed)')
```

And change the network-build call:

```python
    if args.scenario:
        net = build_network(scenario=args.scenario)
    else:
        net = build_network(args.deployment, args.num_nodes, args.seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n base python tests/test_scenarios.py`
Expected: PASS — `build_network(scenario=...) OK`.

- [ ] **Step 5: Checkpoint (no commit)**

Run the full `tests/test_scenarios.py`; confirm green. No commit.

---

### Task 1.3: `freeze_scenarios.py`

**Files:**
- Create: `freeze_scenarios.py`
- Create (dir): `scenarios/`

- [ ] **Step 1: Write the implementation**

```python
# freeze_scenarios.py
"""Snapshot the random deployment scenarios to scenarios/*.csv.

Mirrors benchmark.py's sweep grid. Re-running is idempotent (existing files are
skipped). After freezing, benchmark.py consumes these CSVs instead of
regenerating.

Run:  conda run -n base python freeze_scenarios.py
"""
import os

import numpy as np

from main import build_network
from scenarios import save_scenario

DEPLOYMENTS = ['poisson', 'uniform', 'grid', 'gaussian', 'edge']
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
NUM_NODES = 200
DEPLOYMENT_NODES = {'gaussian': 100}   # match benchmark.py overrides
SCENARIO_DIR = 'scenarios'


def scenario_path(deployment, num_nodes, seed):
    return os.path.join(SCENARIO_DIR, f'{deployment}_n{num_nodes}_s{seed}.csv')


def freeze_one(deployment, num_nodes, seed):
    path = scenario_path(deployment, num_nodes, seed)
    if os.path.exists(path):
        return path, True
    net = build_network(deployment, num_nodes, seed)
    positions = np.array([[s.x, s.y] for s in net.sensors])
    vpre = np.array([s.Vpre for s in net.sensors])
    save_scenario(path, positions, vpre, (0.0, 0.0),
                  {'source': deployment, 'num_nodes': num_nodes, 'seed': seed,
                   'area': net.area, 'coverage_radius': ''})
    return path, False


def main():
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    for deployment in DEPLOYMENTS:
        n = DEPLOYMENT_NODES.get(deployment, NUM_NODES)
        for seed in SEEDS:
            path, skipped = freeze_one(deployment, n, seed)
            print(f"{'skip' if skipped else 'wrote'} {path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it**

Run: `conda run -n base python freeze_scenarios.py`
Expected: 50 lines `wrote scenarios/<deployment>_n<N>_s<seed>.csv` (gaussian at n100, others n200). A second run prints `skip` for all.

- [ ] **Step 3: Spot-check a frozen file reproduces its network**

Run:
```bash
conda run -n base python -c "
import numpy as np
from main import build_network
a = build_network('uniform', 200, 7)
b = build_network(scenario='scenarios/uniform_n200_s7.csv')
pa = np.array([[s.x, s.y] for s in a.sensors]); pb = np.array([[s.x, s.y] for s in b.sensors])
print('positions match:', np.allclose(pa, pb))
print('vpre match:', np.allclose([s.Vpre for s in a.sensors], [s.Vpre for s in b.sensors]))
"
```
Expected: `positions match: True` and `vpre match: True`.

- [ ] **Step 4: Checkpoint (no commit)**

Confirm step 3 prints both `True`. No commit.

---

### Task 1.4: `benchmark.py` prefers frozen scenarios

**Files:**
- Modify: `benchmark.py` (`run_one`)

- [ ] **Step 1: Modify `run_one` to prefer a frozen CSV**

In `benchmark.py`, add this import inside `run_one` next to the existing `from main import ...`:

```python
    from main import build_network, make_algo
    from freeze_scenarios import scenario_path
```

Replace the `net = build_network(deployment, num_nodes, seed)` line with:

```python
        frozen = scenario_path(deployment, num_nodes, seed)
        if os.path.exists(frozen):
            net = build_network(scenario=frozen)
        else:
            net = build_network(deployment, num_nodes, seed)
```

- [ ] **Step 2: Smoke-test one frozen run**

Run:
```bash
conda run -n base python -c "
from benchmark import run_one
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    p = run_one('LEACH', 'uniform', 7, 200, d, max_rounds=50)
    print('wrote', os.path.basename(p), 'exists:', os.path.exists(p))
"
```
Expected: prints `wrote LEACH_uniform_7.json exists: True` and (because `scenarios/uniform_n200_s7.csv` exists from Task 1.3) it used the frozen file.

- [ ] **Step 3: Checkpoint (no commit)**

Confirm the JSON was written. No commit.

---

# PHASE 2 — Base station as a parameter

> Run `gitnexus_impact({target: "calc_node_cost", direction: "upstream"})` and
> `gitnexus_impact({target: "build_routing_tree", direction: "upstream"})` before
> editing `model.py`; report the blast radius. Expect HIGH risk (all algorithms
> depend on the energy model). Default `bs_pos=(0,0)` keeps everything backward
> compatible.

### Task 2.1: `NetworkModel.bs_pos` + `dist_to_bs`

**Files:**
- Modify: `model.py` (`__init__`, new method)
- Test: `tests/test_bs_param.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bs_param.py
"""The base station is a configurable NetworkModel parameter."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Sensor, NetworkModel


def _net(bs_pos):
    sensors = [Sensor(id=0, x=30.0, y=40.0, e0=0.005, power=1e-4, Vpre=3.0)]
    return NetworkModel(sensors, 250, bs_pos=bs_pos)


def test_default_bs_is_origin():
    net = _net((0.0, 0.0))
    assert net.bs_x == 0.0 and net.bs_y == 0.0
    assert math.isclose(net.dist_to_bs(net.sensors[0]), 50.0), "3-4-5 to origin"
    print("  default origin OK")


def test_offset_bs():
    net = _net((30.0, 40.0))
    assert math.isclose(net.dist_to_bs(net.sensors[0]), 0.0), "node sits on BS"
    print("  offset BS OK")


if __name__ == '__main__':
    test_default_bs_is_origin()
    test_offset_bs()
    print("\nAll bs-param tests passed.")
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n base python tests/test_bs_param.py`
Expected: FAIL — `AttributeError: 'NetworkModel' object has no attribute 'bs_x'`.

- [ ] **Step 3: Implement**

In `model.py` `NetworkModel.__init__`, after `self.area = area` (line ~71) add:

```python
        bs_pos = params.get('bs_pos', (0.0, 0.0))
        self.bs_x = float(bs_pos[0])
        self.bs_y = float(bs_pos[1])
```

Add the helper method (place it just above `calc_comm_range`):

```python
    def dist_to_bs(self, sensor):
        """Euclidean distance from a sensor to the base station."""
        return math.hypot(sensor.x - self.bs_x, sensor.y - self.bs_y)
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n base python tests/test_bs_param.py`
Expected: PASS — both lines OK.

- [ ] **Step 5: Checkpoint (no commit)**

---

### Task 2.2: Replace origin-distance sites in `model.py`

**Files:**
- Modify: `model.py` (`calc_node_cost`, `build_routing_tree`, `check_potential_connectivity`)

- [ ] **Step 1: Replace the three internal sites**

In `calc_node_cost` (the `if clustering:` branch, ~line 162):

```python
        if clustering:
            d = self.dist_to_bs(sensor)
```

In `build_routing_tree` gateway test (~line 201):

```python
            if self.dist_to_bs(s) <= s.rc:
```

In `build_routing_tree` root tx distance (~line 239):

```python
                tx_d = self.dist_to_bs(s)
```

- [ ] **Step 2: Make `check_potential_connectivity` include the BS**

The current check verifies node-to-node connectivity only. With an off-centre BS, a node-connected layout can still be unable to reach the BS. After the BFS, add a BS-reachability guard. Replace the final `return bool(seen.all())` with:

```python
        if not seen.all():
            return False
        # At least one node must reach the BS at max power.
        max_rc = self.calc_comm_range(self.p_max)
        d_bs = np.hypot(coords[:, 0] - self.bs_x, coords[:, 1] - self.bs_y)
        return bool(np.any(d_bs <= max_rc))
```

- [ ] **Step 3: Regression — existing behaviour unchanged at origin**

Run:
```bash
conda run -n base python -c "
from main import build_network
net = build_network('uniform', 200, 7)
print('connectable:', net.check_potential_connectivity())
"
```
Expected: `connectable: True` (origin BS, dense layout — unchanged).

- [ ] **Step 4: Checkpoint (no commit)**

Run `conda run -n base python tests/test_bs_param.py` and the regression in step 3. No commit.

---

### Task 2.3: Replace origin-distance sites across all algorithms

**Files:**
- Modify: `algos/gt2.py`, `algos/leach.py`, `algos/gtfr.py`, `algos/ee_tcm.py`, `algos/fl_leach_pso.py`, `algos/sca_levy.py`, `algos/fc_cra.py`

Every `math.hypot(<node>.x, <node>.y)` in the algorithms is a *distance-to-BS*
computation and must become `net.dist_to_bs(<node>)`. **Do not** change hypots
that subtract another point (e.g. `math.hypot(s.x - cx, s.y - cy)` in
`fl_leach_pso.py` is node-to-cluster-centre — leave it).

- [ ] **Step 1: Find every candidate site**

Run:
```bash
grep -rn "hypot([a-z_]*\.x, [a-z_]*\.y)" algos/
```
Expected hits (origin-distance — all must be replaced):
`gt2.py`, `leach.py`, `gtfr.py`, `ee_tcm.py` (×2), `fl_leach_pso.py` (the `dist_bs` one), `sca_levy.py` (×4), `fc_cra.py` (×5, incl. the `np.array([... for s in alive])` and the `s → BS` edge).

- [ ] **Step 2: Apply the transformation per file**

For each hit, the local variable holding the `NetworkModel` is `net` (in the
`_maintenance`/routing helpers) or `self.net` (check the surrounding method —
algorithms store the model as `self.net`). Use whichever name is in scope.

Examples of the exact edits:

`algos/gt2.py` (~line 371):
```python
                tx_dist = info['tx_dist'] if info else net.dist_to_bs(s)
```

`algos/leach.py` (~line 256), `algos/gtfr.py` (~485), `algos/ee_tcm.py` (~477),
`algos/fl_leach_pso.py` (~716): same `... if info else net.dist_to_bs(s)` shape.

`algos/ee_tcm.py` (~line 124):
```python
            e_to_sink = net.calc_tx_cost(net.dist_to_bs(s), 'CH')
```

`algos/fl_leach_pso.py` (~line 279):
```python
                dist_bs = net.dist_to_bs(s)
```

`algos/sca_levy.py`: line ~450 `return net.dist_to_bs(s)`; ~498 `... else net.dist_to_bs(s)`; ~530 `d_ch_bs = net.dist_to_bs(ch)`; ~542 `d_r_bs = net.dist_to_bs(s)`.

`algos/fc_cra.py`: line ~156 `d_bs = np.array([net.dist_to_bs(s) for s in alive], dtype=float)`; ~431 `return net.dist_to_bs(s)`; ~473 `... else net.dist_to_bs(s)`; ~563 `... and net.dist_to_bs(s) <= self._d_max]`; ~606-607 `d = net.dist_to_bs(s)`.

> If a method uses `self.net`, write `self.net.dist_to_bs(...)`. Verify the
> attribute name by reading the method header before editing.

- [ ] **Step 3: Verify no origin-distance hypots remain in algos**

Run:
```bash
grep -rn "hypot([a-z_]*\.x, [a-z_]*\.y)" algos/
```
Expected: **no output** (all replaced; the surviving hypots all subtract a second point).

- [ ] **Step 4: Regression — every algorithm still runs a few rounds**

Run:
```bash
conda run -n base python -c "
import os; os.environ['MPLBACKEND']='Agg'
from main import build_network, make_algo, ALGO_CHOICES
for name in ALGO_CHOICES:
    net = build_network('uniform', 60, 3)
    algo = make_algo(name, net, dict(max_rounds=5, plot_period=10**9))
    algo.run()
    print(f'{name}: t_no_dead={algo.t_no_dead} t={algo.t}')
"
```
Expected: one line per algorithm, no exceptions.

- [ ] **Step 5: Checkpoint (no commit)**

Run the existing smoke suite: `conda run -n base python tests/test_benchmark_smoke.py`. Confirm green. No commit.

---

### Task 2.4: Thread `bs_pos` from scenario through `main.py`

**Files:**
- Modify: `main.py` (`build_network` generate branch + scenario branch)

- [ ] **Step 1: Pass `bs_pos` in both NetworkModel constructions**

In `build_network`'s scenario branch (Task 1.2) the `bs_pos=bs_pos` kwarg is
already present — keep it. In the *generate* branch, add `bs_pos=(0.0, 0.0)` to
the `NetworkModel(...)` kwargs for explicitness:

```python
                           sensor_sample_bits=SENSOR_SAMPLE_BITS,
                           bs_pos=(0.0, 0.0))
```

- [ ] **Step 2: Verify an off-centre frozen scenario loads with its BS**

Run:
```bash
conda run -n base python -c "
import numpy as np, tempfile, os
from main import build_network
from scenarios import save_scenario
net = build_network('uniform', 40, 1)
pos = np.array([[s.x,s.y] for s in net.sensors]); vp=np.array([s.Vpre for s in net.sensors])
with tempfile.TemporaryDirectory() as d:
    p=os.path.join(d,'s.csv')
    save_scenario(p, pos, vp, (100.0,-50.0), {'source':'uniform','num_nodes':40,'seed':1,'area':250,'coverage_radius':''})
    n2=build_network(scenario=p)
    print('bs:', n2.bs_x, n2.bs_y)
"
```
Expected: `bs: 100.0 -50.0`.

- [ ] **Step 3: Checkpoint (no commit)**

---

# PHASE 3 — Coverage-maximising deployment

> Requires `shapely`. Install: `conda install -n base -c conda-forge shapely`
> (or `pip install shapely`). Add it to the README's required-packages list.

### Task 3.1: `regions.py` — geometry + validation

**Files:**
- Create: `regions.py`
- Test: `tests/test_regions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regions.py
"""Region geometry: deployable area, obstacle exclusion, path constraint, BS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regions import Region


def _square_obstacle_def():
    return {
        'area': 250, 'num_nodes': 10, 'coverage_radius': 40, 'seed': 0,
        'bs': [-200.0, -200.0],
        'obstacles': [[[-50, -50], [50, -50], [50, 50], [-50, 50]]],
        'paths': [],
    }


def test_obstacle_excluded():
    r = Region.from_def(_square_obstacle_def())
    assert not r.contains((0.0, 0.0)), "origin is inside the obstacle"
    assert r.contains((-200.0, -200.0)), "corner is deployable"
    print("  obstacle exclusion OK")


def test_bs_in_obstacle_rejected():
    d = _square_obstacle_def()
    d['bs'] = [0.0, 0.0]
    try:
        Region.from_def(d)
    except ValueError:
        print("  BS-in-obstacle rejected OK")
        return
    raise AssertionError("expected ValueError for BS inside obstacle")


def test_path_constrains_placement():
    d = {'area': 250, 'num_nodes': 10, 'coverage_radius': 20, 'seed': 0,
         'bs': [-200.0, -200.0], 'obstacles': [],
         'paths': [{'coords': [[-200, -200], [200, 200]], 'width': 20}]}
    r = Region.from_def(d)
    assert r.contains((0.0, 0.0)), "on the diagonal path"
    assert not r.contains((200.0, -200.0)), "off the path"
    print("  path constraint OK")


if __name__ == '__main__':
    test_obstacle_excluded()
    test_bs_in_obstacle_rejected()
    test_path_constrains_placement()
    print("\nAll region tests passed.")
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n base python tests/test_regions.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'regions'`.

- [ ] **Step 3: Implement**

```python
# regions.py
"""Polygonal region geometry for coverage-deployment scenarios.

A Region is the deployable area for nodes: the square bounds, minus obstacle
polygons, optionally restricted to buffered path corridors. Built from a plain
dict (loaded from a YAML definition).
"""
import numpy as np
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import nearest_points, unary_union


class Region:
    def __init__(self, area, deployable, bs_pos):
        self.area = area
        self.deployable = deployable          # shapely (multi)polygon
        self.bs_pos = (float(bs_pos[0]), float(bs_pos[1]))
        if not self.deployable.contains(Point(self.bs_pos)):
            raise ValueError(
                f"Base station {self.bs_pos} is not in the deployable area "
                f"(inside an obstacle or off the paths).")

    @classmethod
    def from_def(cls, d):
        area = float(d['area'])
        bounds = box(-area, -area, area, area)

        obstacles = [Polygon(ring) for ring in d.get('obstacles', [])]
        obstacle_union = unary_union(obstacles) if obstacles else None

        paths = d.get('paths', [])
        if paths:
            corridors = [LineString(p['coords']).buffer(float(p['width']) / 2.0)
                         for p in paths]
            deployable = unary_union(corridors).intersection(bounds)
        else:
            deployable = bounds

        if obstacle_union is not None:
            deployable = deployable.difference(obstacle_union)

        return cls(area, deployable, d['bs'])

    def contains(self, pt):
        return self.deployable.contains(Point(float(pt[0]), float(pt[1])))

    def sample_inside(self, n, rng):
        """Rejection-sample n points uniformly inside the deployable area."""
        minx, miny, maxx, maxy = self.deployable.bounds
        out = []
        while len(out) < n:
            xs = rng.uniform(minx, maxx, size=n)
            ys = rng.uniform(miny, maxy, size=n)
            for x, y in zip(xs, ys):
                if self.deployable.contains(Point(x, y)):
                    out.append((x, y))
                    if len(out) == n:
                        break
        return np.array(out, dtype=float)

    def project(self, pt):
        """Nearest point inside the deployable area (identity if already in)."""
        p = Point(float(pt[0]), float(pt[1]))
        if self.deployable.contains(p):
            return (p.x, p.y)
        nearest = nearest_points(self.deployable, p)[0]
        return (nearest.x, nearest.y)

    def coverage_targets(self, spacing):
        """A fixed grid of points inside the deployable area, for scoring."""
        minx, miny, maxx, maxy = self.deployable.bounds
        xs = np.arange(minx, maxx + spacing, spacing)
        ys = np.arange(miny, maxy + spacing, spacing)
        pts = [(x, y) for x in xs for y in ys
               if self.deployable.contains(Point(x, y))]
        return np.array(pts, dtype=float)
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n base python tests/test_regions.py`
Expected: PASS — three OK lines.

- [ ] **Step 5: Checkpoint (no commit)**

---

### Task 3.2: `config/coverage.yaml` + projected-PSO placement

**Files:**
- Create: `config/coverage.yaml`
- Create: `coverage_deploy.py`
- Test: `tests/test_coverage_deploy.py`

- [ ] **Step 1: Create the config**

```yaml
# config/coverage.yaml — projected-PSO coverage placement knobs
pso_particles: 30
pso_iters: 80
pso_w: 0.7          # inertia
pso_c1: 1.5         # cognitive
pso_c2: 1.5         # social
penalty_lambda: 2.0 # weight on disconnected-fraction in fitness
target_spacing: 15  # coverage-target grid spacing (m)
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_coverage_deploy.py
"""Projected-PSO coverage placement produces a connected, in-region layout."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regions import Region
from coverage_deploy import place_nodes


def _open_region():
    return Region.from_def({
        'area': 100, 'num_nodes': 12, 'coverage_radius': 40, 'seed': 1,
        'bs': [0.0, 0.0], 'obstacles': [], 'paths': [],
    })


def test_placement_in_region_and_connected():
    region = _open_region()
    positions = place_nodes(region, num_nodes=12, coverage_radius=40,
                            r_conn=60.0, seed=1, config_path='config/coverage.yaml')
    assert positions.shape == (12, 2), "wrong node count"
    for p in positions:
        assert region.contains(p), f"node {p} outside deployable area"

    # connectivity at r_conn including the BS
    from coverage_deploy import is_connected_to_bs
    assert is_connected_to_bs(positions, region.bs_pos, 60.0), "not BS-connected"
    print("  placement in-region + connected OK")


def test_determinism():
    region = _open_region()
    a = place_nodes(region, 12, 40, 60.0, seed=5, config_path='config/coverage.yaml')
    b = place_nodes(region, 12, 40, 60.0, seed=5, config_path='config/coverage.yaml')
    assert np.allclose(a, b), "same seed must give same layout"
    print("  determinism OK")


if __name__ == '__main__':
    test_placement_in_region_and_connected()
    test_determinism()
    print("\nAll coverage-deploy tests passed.")
```

- [ ] **Step 3: Run to verify it fails**

Run: `conda run -n base python tests/test_coverage_deploy.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'coverage_deploy'`.

- [ ] **Step 4: Implement**

```python
# coverage_deploy.py
"""Projected-PSO node placement maximising coverage under a BS-connectivity
constraint, with a hard connectivity-repair post-process.

Decision variables are the N node positions (2*N continuous dims). Particles are
warm-started inside the deployable area and projected back into it after each
update, so they never leave the feasible region. Fitness rewards coverage and
penalises disconnection; a final repair guarantees a BS-connected layout.
"""
import collections

import numpy as np
import yaml


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def is_connected_to_bs(positions, bs_pos, r_conn):
    """True iff every node is in the BS-rooted component of the r_conn graph."""
    n = len(positions)
    pts = np.vstack([np.asarray(bs_pos, dtype=float), positions])  # index 0 = BS
    seen = np.zeros(n + 1, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        nbrs = np.nonzero((d <= r_conn) & ~seen)[0]
        seen[nbrs] = True
        stack.extend(nbrs.tolist())
    return bool(seen[1:].all())


def _coverage_fraction(positions, targets, coverage_radius):
    if len(targets) == 0:
        return 0.0
    # covered if any node within coverage_radius
    covered = np.zeros(len(targets), dtype=bool)
    for x, y in positions:
        d = np.hypot(targets[:, 0] - x, targets[:, 1] - y)
        covered |= d <= coverage_radius
    return float(covered.mean())


def _disconnected_fraction(positions, bs_pos, r_conn):
    n = len(positions)
    pts = np.vstack([np.asarray(bs_pos, dtype=float), positions])
    seen = np.zeros(n + 1, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        nbrs = np.nonzero((d <= r_conn) & ~seen)[0]
        seen[nbrs] = True
        stack.extend(nbrs.tolist())
    return float((~seen[1:]).mean())


def _fitness(positions, targets, coverage_radius, bs_pos, r_conn, lam):
    return (_coverage_fraction(positions, targets, coverage_radius)
            - lam * _disconnected_fraction(positions, bs_pos, r_conn))


def _project_all(region, flat):
    """Project a flat (2N,) vector of positions back into the deployable area."""
    pos = flat.reshape(-1, 2)
    return np.array([region.project(p) for p in pos], dtype=float).ravel()


def _repair_connectivity(positions, region, bs_pos, r_conn, max_iter=1000):
    """Snap disconnected nodes toward the BS component until all are connected."""
    pos = positions.copy()
    for _ in range(max_iter):
        if is_connected_to_bs(pos, bs_pos, r_conn):
            return pos
        n = len(pos)
        anchors = np.vstack([np.asarray(bs_pos, dtype=float), pos])
        seen = np.zeros(n + 1, dtype=bool)
        seen[0] = True
        stack = [0]
        while stack:
            i = stack.pop()
            d = np.hypot(anchors[:, 0] - anchors[i, 0], anchors[:, 1] - anchors[i, 1])
            nbrs = np.nonzero((d <= r_conn) & ~seen)[0]
            seen[nbrs] = True
            stack.extend(nbrs.tolist())
        comp_pts = anchors[seen]              # connected component (incl. BS)
        # pick the disconnected node nearest the component, pull it toward its
        # nearest component point to just within r_conn, then project to region.
        disc_idx = [j for j in range(n) if not seen[j + 1]]
        best = None
        for j in disc_idx:
            d = np.hypot(comp_pts[:, 0] - pos[j, 0], comp_pts[:, 1] - pos[j, 1])
            k = int(np.argmin(d))
            if best is None or d[k] < best[2]:
                best = (j, comp_pts[k], d[k])
        j, anchor, dist = best
        direction = (pos[j] - anchor)
        norm = np.hypot(*direction) or 1.0
        target = anchor + direction / norm * (0.9 * r_conn)
        pos[j] = np.array(region.project(target))
    raise RuntimeError(
        "Could not repair connectivity — N too small or region too sparse for "
        f"r_conn={r_conn}.")


def place_nodes(region, num_nodes, coverage_radius, r_conn, seed, config_path):
    cfg = _load_cfg(config_path)
    rng = np.random.default_rng(seed)
    targets = region.coverage_targets(cfg['target_spacing'])
    lam = cfg['penalty_lambda']

    dim = 2 * num_nodes
    P = cfg['pso_particles']

    # warm-start particles inside the deployable area
    X = np.array([region.sample_inside(num_nodes, rng).ravel() for _ in range(P)])
    V = np.zeros((P, dim))

    pbest = X.copy()
    pbest_fit = np.array([
        _fitness(x.reshape(-1, 2), targets, coverage_radius, region.bs_pos, r_conn, lam)
        for x in X])
    g = int(np.argmax(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = pbest_fit[g]

    w, c1, c2 = cfg['pso_w'], cfg['pso_c1'], cfg['pso_c2']
    for _ in range(cfg['pso_iters']):
        r1 = rng.random((P, dim))
        r2 = rng.random((P, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        for i in range(P):
            X[i] = _project_all(region, X[i])
            fit = _fitness(X[i].reshape(-1, 2), targets, coverage_radius,
                           region.bs_pos, r_conn, lam)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest = X[i].copy()

    positions = gbest.reshape(-1, 2)
    positions = _repair_connectivity(positions, region, region.bs_pos, r_conn)
    return positions
```

- [ ] **Step 5: Run to verify it passes**

Run: `conda run -n base python tests/test_coverage_deploy.py`
Expected: PASS — `placement in-region + connected OK` and `determinism OK`.
(If `determinism` flakes, confirm the single shared `rng` is the only randomness
source — it is; no `np.random.*` global calls.)

- [ ] **Step 6: Checkpoint (no commit)**

---

### Task 3.3: `make_coverage_scenario.py` driver + example region

**Files:**
- Create: `scenarios/defs/example.yaml`
- Create: `make_coverage_scenario.py`

- [ ] **Step 1: Create an example region definition**

```yaml
# scenarios/defs/example.yaml — an L-shaped corridor around a central obstacle
name: example
area: 250
num_nodes: 60
coverage_radius: 45
seed: 7
bs: [-200.0, -200.0]
obstacles:
  - [[-60, -60], [60, -60], [60, 60], [-60, 60]]
paths: []
```

- [ ] **Step 2: Write the driver**

```python
# make_coverage_scenario.py
"""Build a coverage scenario CSV from a region-definition YAML.

Run:  conda run -n base python make_coverage_scenario.py --def scenarios/defs/example.yaml
Output: scenarios/<name>.csv (consumed by build_network(scenario=...)).
"""
import argparse
import os

import numpy as np
import yaml

from regions import Region
from coverage_deploy import place_nodes
from scenarios import save_scenario
from main import (P_MAX, AREA, SNR, NF_RX, N0, BW, WAVE, GAMMA, G_ANT, ETA,
                  R_BIT, E0)
from model import Sensor, NetworkModel

SCENARIO_DIR = 'scenarios'


def _r_conn():
    """Physics max-power comm range — the connectivity radius for validation."""
    # Build a throwaway 1-node net just to call calc_comm_range(p_max).
    s = Sensor(id=0, x=0.0, y=0.0, e0=E0, power=P_MAX, Vpre=3.0)
    net = NetworkModel([s], AREA, snr=SNR, nf_rx=NF_RX, n0=N0, bw=BW, wave=WAVE,
                       gamma=GAMMA, g_ant=G_ANT, eta=ETA, r_bit=R_BIT, p_max=P_MAX)
    return net.calc_comm_range(P_MAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--def', dest='defn', required=True,
                    help='path to region-definition YAML')
    args = ap.parse_args()

    with open(args.defn) as f:
        d = yaml.safe_load(f)

    region = Region.from_def(d)
    r_conn = _r_conn()
    rng = np.random.default_rng(d['seed'])

    positions = place_nodes(region, d['num_nodes'], d['coverage_radius'],
                            r_conn, d['seed'], 'config/coverage.yaml')
    vpre = rng.uniform(2.7, 4.2, size=len(positions))

    os.makedirs(SCENARIO_DIR, exist_ok=True)
    out = os.path.join(SCENARIO_DIR, f"{d['name']}.csv")
    save_scenario(out, positions, vpre, region.bs_pos,
                  {'source': d['name'], 'num_nodes': len(positions),
                   'seed': d['seed'], 'area': d['area'],
                   'coverage_radius': d['coverage_radius']})
    print(f"wrote {out} ({len(positions)} nodes, r_conn={r_conn:.1f} m)")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run the driver end-to-end**

Run: `conda run -n base python make_coverage_scenario.py --def scenarios/defs/example.yaml`
Expected: `wrote scenarios/example.csv (60 nodes, r_conn=99.2 m)`.

- [ ] **Step 4: Confirm the scenario loads and is connectable**

Run:
```bash
conda run -n base python -c "
from main import build_network
net = build_network(scenario='scenarios/example.csv')
print('bs:', net.bs_x, net.bs_y, 'nodes:', net.num_nodes)
print('connectable:', net.check_potential_connectivity())
"
```
Expected: `bs: -200.0 -200.0 nodes: 60` and `connectable: True`.

- [ ] **Step 5: Run one algorithm on the coverage scenario**

Run:
```bash
conda run -n base python -c "
import os; os.environ['MPLBACKEND']='Agg'
from main import build_network, make_algo
net = build_network(scenario='scenarios/example.csv')
algo = make_algo('LEACH', net, dict(max_rounds=50, plot_period=10**9))
algo.run()
print('ran on coverage scenario: t_no_dead=', algo.t_no_dead)
"
```
Expected: prints a round number, no exception.

- [ ] **Step 6: Checkpoint (no commit)**

Run `tests/test_regions.py` and `tests/test_coverage_deploy.py`; confirm green. No commit.

---

### Task 3.4: README — document the new flow

**Files:**
- Modify: `README.md` (or `docs/benchmark.md`)

- [ ] **Step 1: Add a short section**

Document: freezing scenarios (`freeze_scenarios.py`), running with
`--scenario <path>`, authoring a coverage region (`scenarios/defs/*.yaml` →
`make_coverage_scenario.py`), and the new `shapely` dependency.

- [ ] **Step 2: Checkpoint (no commit)**

---

## Self-Review notes (author)

- **Spec coverage:** CSV format → Task 1.1; save/load → 1.1; freeze the five → 1.3; load path in build_network/benchmark → 1.2/1.4; BS parameter + dist_to_bs → 2.1–2.3; BS in connectivity check → 2.2; shapely region/obstacle/path → 3.1; projected PSO + connectivity penalty + repair → 3.2; off-centre BS output → 3.3; testing → tests in each phase; shapely dependency → Phase 3 preamble + 3.4. All spec sections mapped.
- **Type consistency:** `place_nodes(region, num_nodes, coverage_radius, r_conn, seed, config_path)` and `is_connected_to_bs(positions, bs_pos, r_conn)` are used identically in tests and driver. `Region.from_def`, `.contains`, `.sample_inside`, `.project`, `.coverage_targets`, `.bs_pos` consistent across `regions.py`, tests, and `coverage_deploy.py`. `save_scenario(path, positions, vpre, bs_pos, meta)` / `load_scenario(path) -> (positions, vpre, bs_pos, meta)` consistent across all callers.
- **Phase independence:** Phase 1's `bs_pos=bs_pos` kwarg is guarded with a note for the case where Phase 2 isn't merged yet; Phase 2 Task 2.4 re-confirms threading.
```
