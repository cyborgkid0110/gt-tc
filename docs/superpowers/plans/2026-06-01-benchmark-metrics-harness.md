# Benchmark Metrics Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fair, shared metrics-collection layer plus a parallel multi-deployment sweep runner and a plotting module, so GT2 can be benchmarked against the other WSN algorithms on lifetime, energy, data-delivery, and topology metrics.

**Architecture:** A `MetricsCollector` (pure logic) records per-round universal-core metrics by reading shared `NetworkModel` state; `BaseAlgorithm.run()` drives it and dispatches family-specific extras via a `family` class attribute. A standalone `benchmark.py` runs every `(algo, deployment, seed)` combination across independent worker processes, persisting one JSON per run; `plot_benchmark.py` renders figures from those JSON files. The shared energy model and `build_routing_tree()` are reused unchanged — algorithms are measured identically.

**Tech Stack:** Python 3, numpy, matplotlib, `concurrent.futures.ProcessPoolExecutor`, stdlib `json`/`csv`. Conda `base` env (`conda run -n base python ...`).

**Spec:** `docs/superpowers/specs/2026-06-01-benchmark-metrics-harness-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `metrics.py` (create) | `MetricsCollector` class + `clustering_family_metrics(net)` / `topology_family_metrics(net)` helpers. Pure logic; imports only numpy. |
| `algos/__init__.py` (modify) | `BaseAlgorithm`: create `self.metrics`, drive `record_round`/`finalize` in `run()`, add `family` class attr + `_collect_family_metrics()` dispatch. |
| `algos/{gt2,leach,gtfr,fl_leach_pso,sca_levy,fc_cra,ee_tcm}.py` (modify) | Add `family = 'clustering'` class attribute. |
| `algos/{dia_mia,tcle,eftcg}.py` (modify) | Add `family = 'topology'` class attribute. |
| `benchmark.py` (create) | Parallel sweep runner: `run_one(...)` worker, `build_summary_csv(...)`, `main()`. Reuses `main.build_network`/`make_algo`. |
| `plot_benchmark.py` (create) | `load_runs`, survival/energy curves, bar charts, `generate_all(results_dir)`. Reads `results/`, writes `results/figures/`. |
| `tests/test_metrics.py` (create) | Unit tests for `MetricsCollector` + family helpers (fake net). |
| `tests/test_metrics_integration.py` (create) | Runs LEACH a few rounds; checks metrics populated + `fnd == t_no_dead`; topology dispatch. |
| `tests/test_benchmark_smoke.py` (create) | `run_one` writes JSON; `build_summary_csv` writes CSV. |
| `tests/test_plot_smoke.py` (create) | `generate_all` produces PNGs from synthetic run files. |
| `results/` (generated) | `runs/<algo>_<deploy>_<seed>.json`, `summary.csv`, `figures/*.png`. |

---

## Task 1: `MetricsCollector` + family helpers (pure logic, TDD)

**Files:**
- Create: `metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
"""Unit tests for metrics.MetricsCollector and family helpers.

Run:  conda run -n base python tests/test_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    MetricsCollector,
    clustering_family_metrics,
    topology_family_metrics,
)


class FakeSensor:
    def __init__(self, id, e0):
        self.id = id
        self.e0 = e0
        self.e_res = e0
        self.is_ch = False
        self.ch_belong = None
        self.power = 0.0

    @property
    def is_alive(self):
        return self.e_res > 0


class FakeNet:
    """Minimal stand-in for NetworkModel used by MetricsCollector."""
    def __init__(self, sensors):
        self.sensors = sensors
        self.num_nodes = len(sensors)
        self.edges = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        self._reachable = set(s.id for s in sensors)

    def build_routing_tree(self):
        # a node "delivers" iff it is alive and currently reachable
        return {sid: {} for sid in self._reachable
                if self.sensors[sid].is_alive}


def test_lifetime_markers():
    sensors = [FakeSensor(i, e0=10.0) for i in range(4)]
    net = FakeNet(sensors)
    mc = MetricsCollector(net)

    mc.record_round(net, 0, {})           # all 4 alive
    sensors[0].e_res = 0.0
    mc.record_round(net, 1, {})           # 3 alive < 4  -> fnd = 1
    sensors[1].e_res = 0.0
    sensors[2].e_res = 0.0
    mc.record_round(net, 2, {})           # 1 alive <= 2 -> hnd = 2

    s = mc.finalize()
    assert s['fnd'] == 1, s
    assert s['hnd'] == 2, s
    assert s['lnd'] == 2, s
    print('  lifetime markers OK')


def test_pdr_and_energy_per_packet():
    sensors = [FakeSensor(i, e0=10.0) for i in range(4)]   # initial total = 40
    net = FakeNet(sensors)
    mc = MetricsCollector(net)

    mc.record_round(net, 0, {})                            # delivered 4 / gen 4
    for s in sensors:
        s.e_res -= 1.0                                     # total alive energy = 36
    net._reachable = {0, 1, 2}                             # node 3 partitioned
    mc.record_round(net, 1, {})                            # delivered 3 / gen 4

    s = mc.finalize()
    assert s['total_delivered'] == 7, s
    assert s['total_generated'] == 8, s
    assert abs(s['cumulative_pdr'] - 7 / 8) < 1e-9, s
    assert abs(s['energy_drained'] - 4.0) < 1e-9, s
    assert abs(s['energy_per_packet'] - 4.0 / 7.0) < 1e-9, s
    print('  pdr / energy-per-packet OK')


def test_energy_per_packet_none_when_no_delivery():
    sensors = [FakeSensor(i, e0=10.0) for i in range(2)]
    net = FakeNet(sensors)
    net._reachable = set()                                 # nobody delivers
    mc = MetricsCollector(net)
    mc.record_round(net, 0, {})
    s = mc.finalize()
    assert s['total_delivered'] == 0, s
    assert s['energy_per_packet'] is None, s
    print('  no-delivery guard OK')


def test_clustering_family_metrics():
    sensors = [FakeSensor(i, 10.0) for i in range(5)]
    sensors[0].is_ch = True
    sensors[1].ch_belong = sensors[0]
    sensors[2].ch_belong = sensors[0]
    net = FakeNet(sensors)
    m = clustering_family_metrics(net)
    assert m['ch_count'] == 1, m
    assert m['cluster_sizes'] == [2], m
    print('  clustering family OK')


def test_topology_family_metrics():
    sensors = [FakeSensor(i, 10.0) for i in range(3)]
    for s in sensors:
        s.power = 1.0e-4
    net = FakeNet(sensors)
    net.edges[0, 1] = 1
    net.edges[0, 2] = 1
    net.edges[1, 0] = 1                                    # degrees: 2, 1, 0 -> avg 1.0
    m = topology_family_metrics(net)
    assert abs(m['avg_degree'] - 1.0) < 1e-9, m
    assert abs(m['avg_tx_power'] - 1.0e-4) < 1e-12, m
    assert m['lambda2'] is None, m
    print('  topology family OK')


if __name__ == '__main__':
    test_lifetime_markers()
    test_pdr_and_energy_per_packet()
    test_energy_per_packet_none_when_no_delivery()
    test_clustering_family_metrics()
    test_topology_family_metrics()
    print('\nAll metrics tests passed.')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n base python tests/test_metrics.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'` (or ImportError for the names).

- [ ] **Step 3: Implement `metrics.py`**

Create `metrics.py`:

```python
"""Benchmark metrics collection (shared, algorithm-agnostic).

A MetricsCollector records per-round universal-core metrics by reading
NetworkModel state, and derives scalar summary metrics at the end of a run.
Family-specific extras are computed by the two helper functions and merged in
by the caller (BaseAlgorithm._collect_family_metrics).
"""
import numpy as np


def clustering_family_metrics(net):
    """Per-round extras for clustering algorithms: CH count + cluster sizes."""
    alive = [s for s in net.sensors if s.is_alive]
    chs = [s for s in alive if s.is_ch]
    sizes = {ch.id: 0 for ch in chs}
    for s in alive:
        ch = s.ch_belong
        if ch is not None and ch.id in sizes:
            sizes[ch.id] += 1
    return {'ch_count': len(chs), 'cluster_sizes': list(sizes.values())}


def topology_family_metrics(net):
    """Per-round extras for topology-control algorithms.

    avg_degree: mean out-degree (to alive nodes) over alive nodes.
    avg_tx_power: mean transmit power over alive nodes.
    lambda2: left None here (optional; not all algorithms expose it).
    """
    alive_mask = np.array([s.is_alive for s in net.sensors])
    alive_ids = [s.id for s in net.sensors if s.is_alive]
    if alive_ids:
        degrees = [int(net.edges[i][alive_mask].sum()) for i in alive_ids]
        avg_degree = float(np.mean(degrees))
        avg_tx_power = float(np.mean([net.sensors[i].power for i in alive_ids]))
    else:
        avg_degree = 0.0
        avg_tx_power = 0.0
    return {'avg_degree': avg_degree, 'avg_tx_power': avg_tx_power,
            'lambda2': None}


class MetricsCollector:
    """Collects per-round metrics for a single simulation run."""

    def __init__(self, net):
        self.num_nodes = net.num_nodes
        self.initial_total_energy = float(sum(s.e0 for s in net.sensors))

        # per-round universal-core series
        self.rounds = []
        self.alive = []
        self.total_energy = []
        self.energy_std = []
        self.delivered = []
        self.generated = []
        # family-extra series: metric_name -> list (one entry per recorded round)
        self.family = {}

        self._summary = None

    def record_round(self, net, t, family_extras=None):
        """Append one round's metrics. Call once per completed round."""
        alive_sensors = [s for s in net.sensors if s.is_alive]
        alive = len(alive_sensors)
        energies = [s.e_res for s in alive_sensors]
        total_e = float(sum(energies))
        e_std = float(np.std(energies)) if alive >= 2 else 0.0
        delivered = len(net.build_routing_tree())

        self.rounds.append(t)
        self.alive.append(alive)
        self.total_energy.append(total_e)
        self.energy_std.append(e_std)
        self.delivered.append(delivered)
        self.generated.append(alive)

        if family_extras:
            for k, v in family_extras.items():
                self.family.setdefault(k, []).append(v)

    def finalize(self):
        """Derive scalar summary metrics. Returns the summary dict."""
        n = self.num_nodes
        fnd = next((r for r, a in zip(self.rounds, self.alive) if a < n), None)
        hnd = next((r for r, a in zip(self.rounds, self.alive)
                    if a <= n / 2), None)
        lnd = self.rounds[-1] if self.rounds else None

        total_delivered = int(sum(self.delivered))
        total_generated = int(sum(self.generated))
        per_round_pdr = [d / g for d, g in zip(self.delivered, self.generated)
                         if g > 0]
        mean_pdr = float(np.mean(per_round_pdr)) if per_round_pdr else 0.0
        cumulative_pdr = (total_delivered / total_generated
                          if total_generated > 0 else 0.0)

        final_energy = (self.total_energy[-1] if self.total_energy
                        else self.initial_total_energy)
        energy_drained = self.initial_total_energy - final_energy
        energy_per_packet = (energy_drained / total_delivered
                             if total_delivered > 0 else None)
        mean_energy_std = (float(np.mean(self.energy_std))
                           if self.energy_std else 0.0)

        self._summary = {
            'fnd': fnd,
            'hnd': hnd,
            'lnd': lnd,
            'total_delivered': total_delivered,
            'total_generated': total_generated,
            'mean_pdr': mean_pdr,
            'cumulative_pdr': cumulative_pdr,
            'energy_drained': energy_drained,
            'energy_per_packet': energy_per_packet,
            'mean_energy_std': mean_energy_std,
        }
        return self._summary

    def summary(self):
        """Scalar summary dict (finalizes lazily if needed)."""
        if self._summary is None:
            self.finalize()
        return self._summary

    def time_series(self):
        """Per-round series as a JSON-serialisable dict."""
        return {
            'rounds': self.rounds,
            'alive': self.alive,
            'total_energy': self.total_energy,
            'energy_std': self.energy_std,
            'delivered': self.delivered,
            'generated': self.generated,
            'family': self.family,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n base python tests/test_metrics.py`
Expected: PASS — prints each `... OK` line and `All metrics tests passed.`

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "feat: add MetricsCollector and family-metric helpers"
```

---

## Task 2: Wire metrics into `BaseAlgorithm` + family attributes

**Files:**
- Modify: `algos/__init__.py`
- Modify: `algos/gt2.py`, `algos/leach.py`, `algos/gtfr.py`, `algos/fl_leach_pso.py`, `algos/sca_levy.py`, `algos/fc_cra.py`, `algos/ee_tcm.py` (clustering)
- Modify: `algos/dia_mia.py`, `algos/tcle.py`, `algos/eftcg.py` (topology)
- Test: `tests/test_metrics_integration.py`

- [ ] **Step 1: Run GitNexus impact analysis (CLAUDE.md requirement)**

Run impact analysis on the two symbols being edited and report the blast radius to the user before editing:

```
gitnexus_impact({target: "run", direction: "upstream"})
gitnexus_impact({target: "__init__", direction: "upstream"})
```

Expected: all 10 algorithm subclasses appear as dependents (they inherit `run()` and call `super().__init__()`). The changes in this task are **additive** — they must not alter existing round semantics or energy use. If impact reports HIGH/CRITICAL risk, warn the user before proceeding.

- [ ] **Step 2: Write the failing integration test**

Create `tests/test_metrics_integration.py`:

```python
"""Integration test: metrics populated after a real run; family dispatch.

Run:  conda run -n base python tests/test_metrics_integration.py
"""
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_network
from algos import BaseAlgorithm
from algos.leach import LEACH


def test_metrics_populated_after_run():
    net = build_network('poisson', 50, 1)
    algo = LEACH(net, max_rounds=3, plot_period=10 ** 9)
    algo.run()

    assert len(algo.metrics.rounds) >= 1, 'no rounds recorded'
    assert algo.metrics.alive[0] <= 50
    s = algo.metrics.summary()
    # fnd uses pre-increment t, same convention as t_no_dead
    assert s['fnd'] == algo.t_no_dead, (s['fnd'], algo.t_no_dead)
    # LEACH is a clustering algorithm -> CH count series present
    assert 'ch_count' in algo.metrics.family, algo.metrics.family.keys()
    print('  metrics populated OK')


def test_family_dispatch_topology():
    net = build_network('poisson', 20, 1)

    class Dummy(BaseAlgorithm):
        family = 'topology'

        def _run_round(self):
            return False

    algo = Dummy(net, config_path='unused')
    extras = algo._collect_family_metrics()
    assert set(extras) == {'avg_degree', 'avg_tx_power', 'lambda2'}, extras
    print('  topology dispatch OK')


def test_family_dispatch_default_empty():
    net = build_network('poisson', 20, 1)

    class Dummy(BaseAlgorithm):
        def _run_round(self):
            return False

    algo = Dummy(net, config_path='unused')
    assert algo._collect_family_metrics() == {}
    print('  default dispatch OK')


if __name__ == '__main__':
    test_metrics_populated_after_run()
    test_family_dispatch_topology()
    test_family_dispatch_default_empty()
    print('\nAll integration tests passed.')
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `conda run -n base python tests/test_metrics_integration.py`
Expected: FAIL — `AttributeError: 'LEACH' object has no attribute 'metrics'` (and `_collect_family_metrics` missing).

- [ ] **Step 4: Modify `algos/__init__.py`**

Replace the entire file with:

```python
from abc import ABC, abstractmethod

from model import NetworkModel
from metrics import (
    MetricsCollector,
    clustering_family_metrics,
    topology_family_metrics,
)


class BaseAlgorithm(ABC):
    """Abstract base for all WSN topology-control algorithms."""

    # Subclasses set this to 'clustering' or 'topology' to opt into
    # family-specific per-round metrics. None => universal core only.
    family = None

    def __init__(self, net: NetworkModel, config_path: str,
                 max_rounds: int = 50000, plot_period: int = 1000):
        self.net = net
        self.max_rounds = max_rounds
        self.plot_period = plot_period

        # simulation bookkeeping
        self.t = 0
        self.t_no_dead = None
        self.dead_nodes = 0

        # metrics collection (shared, algorithm-agnostic)
        self.metrics = MetricsCollector(net)

    @abstractmethod
    def _run_round(self) -> bool:
        """Execute one simulation round. Return False when network is fully dead."""
        ...

    def run(self):
        """Main simulation loop."""
        while self.t < self.max_rounds:
            ok = self._run_round()
            if not ok:
                break
            # record before incrementing t so fnd matches t_no_dead's convention
            self.metrics.record_round(self.net, self.t,
                                      self._collect_family_metrics())
            self.t += 1

        self.metrics.finalize()

        print(f'Iterations without dead node: {self.t_no_dead}')
        print(f'Iterations without alive node: {self.t}')

    def _collect_family_metrics(self) -> dict:
        """Family-specific per-round extras, dispatched on `self.family`."""
        if self.family == 'clustering':
            return clustering_family_metrics(self.net)
        if self.family == 'topology':
            return topology_family_metrics(self.net)
        return {}

    def _track_death(self, sensor):
        """Increment dead count and record first-death round."""
        self.dead_nodes += 1
        if self.t_no_dead is None:
            self.t_no_dead = self.t
        # print('Dead nodes:', self.dead_nodes)
```

- [ ] **Step 5: Add `family = 'clustering'` to the seven clustering algorithms**

In each of these files, add a class attribute line `    family = 'clustering'` (4-space indent) as the first statement inside the class body, immediately above its `def __init__`:

- `algos/gt2.py` (class `GT2`, above `def __init__` at line ~22)
- `algos/leach.py` (class `LEACH`, above `def __init__` at line ~22)
- `algos/gtfr.py` (class `GTFR`, above `def __init__` at line ~27)
- `algos/fl_leach_pso.py` (class `FLLEACHPSO`, above `def __init__` at line ~38)
- `algos/sca_levy.py` (class `SCALEVY`, above `def __init__` at line ~41)
- `algos/fc_cra.py` (class `FCCRA`, above `def __init__` at line ~53)
- `algos/ee_tcm.py` (class `EETCM`, above `def __init__` at line ~48)

Example (gt2.py):

```python
class GT2(BaseAlgorithm):
    """..."""
    family = 'clustering'

    def __init__(self, net: NetworkModel, config_path: str = 'config/gt2.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)
```

- [ ] **Step 6: Add `family = 'topology'` to the three topology algorithms**

Same pattern, value `'topology'`:

- `algos/dia_mia.py` (class `DIAMIA`, above `def __init__` at line ~45)
- `algos/tcle.py` (class `TCLE`, above `def __init__` at line ~42)
- `algos/eftcg.py` (class `EFTCG`, above `def __init__` at line ~45)

Example (dia_mia.py):

```python
class DIAMIA(BaseAlgorithm):
    """..."""
    family = 'topology'

    def __init__(self, net: NetworkModel, config_path: str = 'config/dia_mia.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)
```

- [ ] **Step 7: Run the integration test to verify it passes**

Run: `conda run -n base python tests/test_metrics_integration.py`
Expected: PASS — prints `metrics populated OK`, `topology dispatch OK`, `default dispatch OK`, `All integration tests passed.`

- [ ] **Step 8: Re-run the metrics unit tests (no regression)**

Run: `conda run -n base python tests/test_metrics.py`
Expected: PASS — `All metrics tests passed.`

- [ ] **Step 9: Commit**

```bash
git add algos/__init__.py algos/gt2.py algos/leach.py algos/gtfr.py \
        algos/fl_leach_pso.py algos/sca_levy.py algos/fc_cra.py algos/ee_tcm.py \
        algos/dia_mia.py algos/tcle.py algos/eftcg.py \
        tests/test_metrics_integration.py
git commit -m "feat: drive MetricsCollector from BaseAlgorithm.run + family attrs"
```

---

## Task 3: Parallel sweep runner `benchmark.py`

**Files:**
- Create: `benchmark.py`
- Test: `tests/test_benchmark_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_benchmark_smoke.py`:

```python
"""Smoke test for the parallel sweep runner.

Run:  conda run -n base python tests/test_benchmark_smoke.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark


def test_run_one_writes_json_and_csv():
    d = tempfile.mkdtemp()
    out = benchmark.run_one('LEACH', 'poisson', 1, 40, d, max_rounds=2)
    assert os.path.exists(out), out

    with open(out) as f:
        payload = json.load(f)
    assert payload['algo'] == 'LEACH'
    assert payload['deployment'] == 'poisson'
    assert payload['seed'] == 1
    assert payload['num_nodes'] == 40
    assert 'summary' in payload and 'time_series' in payload
    assert 'fnd' in payload['summary']
    assert 'alive' in payload['time_series']
    print('  run_one JSON OK')

    # second call must skip (idempotent / resumable)
    out2 = benchmark.run_one('LEACH', 'poisson', 1, 40, d, max_rounds=2)
    assert out2 == out
    print('  skip-existing OK')

    benchmark.build_summary_csv(d)
    csv_path = os.path.join(d, 'summary.csv')
    assert os.path.exists(csv_path), csv_path
    with open(csv_path) as f:
        header = f.readline()
    assert 'algo' in header and 'fnd' in header
    print('  summary.csv OK')


if __name__ == '__main__':
    test_run_one_writes_json_and_csv()
    print('\nBenchmark smoke test passed.')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n base python tests/test_benchmark_smoke.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark'`.

- [ ] **Step 3: Implement `benchmark.py`**

Create `benchmark.py`:

```python
"""Parallel multi-deployment benchmark sweep.

Runs every (algorithm, deployment, seed) combination in independent worker
processes, persisting one JSON per run under results/runs/. summary.csv is
rebuilt from those JSON files after the pool drains. Re-running skips existing
run files (resumable).

Run:  conda run -n base python benchmark.py
"""
import contextlib
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

ALGOS = [
    'GT2', 'LEACH', 'GTFR', 'DIA', 'MIA', 'TCLE',
    'EFTCG-1', 'EFTCG-2', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA', 'EE-TCM',
]
DEPLOYMENTS = ['poisson', 'uniform', 'grid', 'gaussian', 'edge']
SEEDS = [1, 2, 3]
NUM_NODES = 200
RESULTS_DIR = 'results'
WORKERS = os.cpu_count()
MAX_ROUNDS = 50000

SUMMARY_FIELDS = [
    'algo', 'deployment', 'seed', 'num_nodes',
    'fnd', 'hnd', 'lnd',
    'total_delivered', 'total_generated', 'mean_pdr', 'cumulative_pdr',
    'energy_drained', 'energy_per_packet', 'mean_energy_std',
]


def run_one(algo, deployment, seed, num_nodes, results_dir,
            max_rounds=MAX_ROUNDS):
    """Run a single (algo, deployment, seed) and write its JSON. Resumable.

    Returns the output path. Skips (and returns the path) if it already exists.
    Picklable top-level function for ProcessPoolExecutor.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')
    runs_dir = os.path.join(results_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    out = os.path.join(runs_dir, f'{algo}_{deployment}_{seed}.json')
    if os.path.exists(out):
        return out

    from main import build_network, make_algo

    log_path = os.path.join(runs_dir, f'{algo}_{deployment}_{seed}.log')
    with open(log_path, 'w') as lf, contextlib.redirect_stdout(lf):
        net = build_network(deployment, num_nodes, seed)
        algo_obj = make_algo(
            algo, net, dict(max_rounds=max_rounds, plot_period=10 ** 9))
        algo_obj.run()
        payload = {
            'algo': algo,
            'deployment': deployment,
            'seed': seed,
            'num_nodes': num_nodes,
            'summary': algo_obj.metrics.summary(),
            'time_series': algo_obj.metrics.time_series(),
        }

    with open(out, 'w') as f:
        json.dump(payload, f)
    return out


def build_summary_csv(results_dir):
    """Rebuild summary.csv from all per-run JSON files (one row per run)."""
    runs_dir = os.path.join(results_dir, 'runs')
    rows = []
    for fn in sorted(os.listdir(runs_dir)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(runs_dir, fn)) as f:
            p = json.load(f)
        row = {'algo': p['algo'], 'deployment': p['deployment'],
               'seed': p['seed'], 'num_nodes': p['num_nodes']}
        row.update(p['summary'])
        rows.append(row)

    csv_path = os.path.join(results_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return csv_path


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'runs'), exist_ok=True)
    tasks = list(product(ALGOS, DEPLOYMENTS, SEEDS))
    total = len(tasks)
    print(f'Dispatching {total} runs across {WORKERS} workers...')

    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, a, d, s, NUM_NODES, RESULTS_DIR): (a, d, s)
                for a, d, s in tasks}
        for fut in as_completed(futs):
            a, d, s = futs[fut]
            try:
                fut.result()
                done += 1
                print(f'[{done}/{total}] done: {a} / {d} / seed {s}')
            except Exception as e:
                print(f'FAILED: {a} / {d} / seed {s}: {e!r}')

    path = build_summary_csv(RESULTS_DIR)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `conda run -n base python tests/test_benchmark_smoke.py`
Expected: PASS — prints `run_one JSON OK`, `skip-existing OK`, `summary.csv OK`, `Benchmark smoke test passed.`

- [ ] **Step 5: Commit**

```bash
git add benchmark.py tests/test_benchmark_smoke.py
git commit -m "feat: add parallel multi-deployment benchmark sweep runner"
```

---

## Task 4: Plotting module `plot_benchmark.py`

**Files:**
- Create: `plot_benchmark.py`
- Test: `tests/test_plot_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_plot_smoke.py`:

```python
"""Smoke test: plot_benchmark.generate_all produces PNG figures.

Run:  conda run -n base python tests/test_plot_smoke.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_benchmark


def _write_run(runs_dir, algo, deployment, seed):
    payload = {
        'algo': algo, 'deployment': deployment, 'seed': seed, 'num_nodes': 4,
        'summary': {
            'fnd': 5, 'hnd': 8, 'lnd': 10,
            'total_delivered': 30, 'total_generated': 40,
            'mean_pdr': 0.9, 'cumulative_pdr': 0.75,
            'energy_drained': 1.0, 'energy_per_packet': 0.03,
            'mean_energy_std': 0.1,
        },
        'time_series': {
            'rounds': [0, 1, 2], 'alive': [4, 4, 3],
            'total_energy': [40.0, 38.0, 36.0], 'energy_std': [0.0, 0.0, 0.1],
            'delivered': [4, 4, 3], 'generated': [4, 4, 3], 'family': {},
        },
    }
    with open(os.path.join(runs_dir, f'{algo}_{deployment}_{seed}.json'),
              'w') as f:
        json.dump(payload, f)


def test_generate_all_creates_pngs():
    d = tempfile.mkdtemp()
    runs_dir = os.path.join(d, 'runs')
    os.makedirs(runs_dir)
    _write_run(runs_dir, 'LEACH', 'poisson', 1)
    _write_run(runs_dir, 'LEACH', 'poisson', 2)
    _write_run(runs_dir, 'GT2', 'poisson', 1)

    plot_benchmark.generate_all(d)

    fig_dir = os.path.join(d, 'figures')
    assert os.path.isdir(fig_dir), fig_dir
    pngs = [fn for fn in os.listdir(fig_dir) if fn.endswith('.png')]
    assert pngs, 'no PNG figures produced'
    assert 'survival_poisson.png' in pngs, pngs
    assert 'fnd.png' in pngs, pngs
    print('  figures produced:', sorted(pngs))


if __name__ == '__main__':
    test_generate_all_creates_pngs()
    print('\nPlot smoke test passed.')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n base python tests/test_plot_smoke.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plot_benchmark'`.

- [ ] **Step 3: Implement `plot_benchmark.py`**

Create `plot_benchmark.py`:

```python
"""Render benchmark figures from persisted run JSON (never re-runs the sweep).

Reads results/runs/*.json and writes results/figures/*.png.

Run:  conda run -n base python plot_benchmark.py
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'results'


def load_runs(results_dir=RESULTS_DIR):
    runs = []
    for fn in glob.glob(os.path.join(results_dir, 'runs', '*.json')):
        with open(fn) as f:
            runs.append(json.load(f))
    return runs


def _aligned_mean_std(series_list):
    """Mean/std across runs of differing length; pad short runs with last value."""
    length = max(len(s) for s in series_list)
    arr = np.full((len(series_list), length), np.nan)
    for i, s in enumerate(series_list):
        if not s:
            arr[i, :] = 0.0
            continue
        arr[i, :len(s)] = s
        arr[i, len(s):] = s[-1]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _curve(runs, deployment, ts_key, ylabel, fname, fig_dir):
    by_algo = defaultdict(list)
    for r in runs:
        if r['deployment'] != deployment:
            continue
        by_algo[r['algo']].append(r['time_series'][ts_key])
    if not by_algo:
        return
    plt.figure(figsize=(8, 5))
    for algo, series_list in sorted(by_algo.items()):
        mean, std = _aligned_mean_std(series_list)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} — {deployment}')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=120)
    plt.close()


def _bar_metric(runs, key, ylabel, fname, fig_dir):
    by_algo = defaultdict(list)
    for r in runs:
        v = r['summary'].get(key)
        if v is None:
            continue
        by_algo[r['algo']].append(v)
    if not by_algo:
        return
    algos = sorted(by_algo)
    means = [np.mean(by_algo[a]) for a in algos]
    stds = [np.std(by_algo[a]) for a in algos]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(algos)), means, yerr=stds, capsize=3)
    plt.xticks(range(len(algos)), algos, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=120)
    plt.close()


def generate_all(results_dir=RESULTS_DIR):
    runs = load_runs(results_dir)
    if not runs:
        print('No runs found; nothing to plot.')
        return
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for deployment in sorted({r['deployment'] for r in runs}):
        _curve(runs, deployment, 'alive', 'Alive nodes',
               f'survival_{deployment}.png', fig_dir)
        _curve(runs, deployment, 'total_energy', 'Total residual energy (J)',
               f'energy_{deployment}.png', fig_dir)

    _bar_metric(runs, 'fnd', 'First Node Death (round)', 'fnd.png', fig_dir)
    _bar_metric(runs, 'hnd', 'Half Node Death (round)', 'hnd.png', fig_dir)
    _bar_metric(runs, 'lnd', 'Last Node Death (round)', 'lnd.png', fig_dir)
    _bar_metric(runs, 'total_delivered', 'Total packets to BS',
                'delivered.png', fig_dir)
    _bar_metric(runs, 'cumulative_pdr', 'Cumulative PDR', 'pdr.png', fig_dir)
    _bar_metric(runs, 'energy_per_packet', 'Energy per delivered packet (J)',
                'energy_per_packet.png', fig_dir)

    print(f'Figures written to {fig_dir}')


if __name__ == '__main__':
    generate_all()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n base python tests/test_plot_smoke.py`
Expected: PASS — prints `figures produced: [...]` including `fnd.png` and `survival_poisson.png`, then `Plot smoke test passed.`

- [ ] **Step 5: Commit**

```bash
git add plot_benchmark.py tests/test_plot_smoke.py
git commit -m "feat: add benchmark plotting module (curves + bar charts)"
```

---

## Task 5: Ignore generated results + docs touch-up

**Files:**
- Modify/Create: `.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `results/` to `.gitignore`**

Append to `.gitignore` (create it if absent):

```
# benchmark sweep outputs (regenerate via benchmark.py / plot_benchmark.py)
results/
```

- [ ] **Step 2: Document the harness in `CLAUDE.md`**

Under the "Running the simulation" section of `CLAUDE.md`, add:

```markdown
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
```

- [ ] **Step 3: Run the full test suite (no regression)**

Run:
```bash
conda run -n base python tests/test_metrics.py
conda run -n base python tests/test_metrics_integration.py
conda run -n base python tests/test_benchmark_smoke.py
conda run -n base python tests/test_plot_smoke.py
```
Expected: each prints its `... passed.` line.

- [ ] **Step 4: Commit**

```bash
git add .gitignore CLAUDE.md
git commit -m "docs: document benchmark harness + ignore results/"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** universal-core metrics → Task 1 `MetricsCollector`; family extras → Task 1 helpers + Task 2 dispatch; connectivity-based delivery via `build_routing_tree()` → Task 1 `record_round`; shared collection in `BaseAlgorithm` → Task 2; parallel sweep + resumable + summary.csv-after-drain → Task 3; plots + CSV/JSON outputs → Tasks 3–4; determinism via `build_network(seed)` → Task 3 worker. All covered.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; every command shows expected output.
- **Type/name consistency:** `MetricsCollector(net)`, `record_round(net, t, family_extras)`, `finalize()`, `summary()`, `time_series()`, `clustering_family_metrics`/`topology_family_metrics`, `run_one(algo, deployment, seed, num_nodes, results_dir, max_rounds)`, `build_summary_csv(results_dir)`, `generate_all(results_dir)` are used identically across tasks and tests. Summary keys (`fnd/hnd/lnd/total_delivered/total_generated/mean_pdr/cumulative_pdr/energy_drained/energy_per_packet/mean_energy_std`) match `SUMMARY_FIELDS` and the plot/CSV consumers.
- **Off-by-one:** `record_round` is called before `t += 1`, so `fnd` (first round with `alive < N`) equals `t_no_dead` (set with pre-increment `t`); asserted in Task 2.
```
