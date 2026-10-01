# Benchmark Results Analysis & Paper Write-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two missing per-round metrics (avg hop-to-BS, universal avg tx-power) to the benchmark harness, re-run the sweep, generate per-scenario figures for the three analysis criteria, and produce a bullet-point writing guideline for paper §5.2–5.4.

**Architecture:** `metrics.py` gains two universal-core per-round series computed in `record_round` (reusing the routing tree already built for `delivered`). `benchmark.py` and `plot_benchmark.py` surface the new scalar summaries and per-scenario figures. The sweep is re-run (resumable) after backing up the prior results. Finally a markdown guideline is written from the regenerated `summary_by_scenario.csv`.

**Tech Stack:** Python 3 (Anaconda `base` env), numpy, matplotlib (Agg backend). Tests are standalone scripts run with `conda run -n base python tests/<file>.py` (no pytest dependency — follow the existing pattern).

**Spec:** `docs/superpowers/specs/2026-06-13-benchmark-results-analysis-and-paper-design.md`

**Environment note:** Every Python command must run inside the `base` conda env: prefix with `conda run -n base`.

---

## File Structure

- `metrics.py` — MODIFY. Add `avg_hop` + universal `avg_tx_power` per-round series; add `mean_avg_hop`, `mean_avg_tx_power` to the summary dict. Refactor `record_round` to build the routing tree once.
- `tests/test_metrics.py` — MODIFY. Update `FakeNet.build_routing_tree` to return `depth`; add `power` handling; add tests for the two new metrics.
- `benchmark.py` — MODIFY. Add the two new scalar fields to `SUMMARY_FIELDS`.
- `plot_benchmark.py` — MODIFY. Add `_family_curve` helper; add per-scenario hop / CH-count / tx-power figures to `generate_all`; add new scalar metrics + a median energy-per-packet column to the scenario summary.
- `tests/test_plot_smoke.py` — MODIFY. Extend the smoke test to exercise the new figure/summary code paths.
- `~/Documents/workspace/github/GT2_paper/sec4_simulation.tex` — MODIFY. Insert bullet-point writing guidelines as LaTeX comments inside §5.2–5.4 (they do not render; the user writes the prose around them).

**Commits:** none. This plan makes **no git commits** — all changes stay in the working tree for the user to review and commit themselves.

---

## Task 1: Add `avg_hop` and universal `avg_tx_power` to `metrics.py`

**Files:**
- Modify: `metrics.py` (`MetricsCollector.__init__`, `record_round`, `finalize`, `time_series`)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Update the test fakes so the routing tree carries `depth`**

In `tests/test_metrics.py`, replace `FakeNet.build_routing_tree` (currently lines ~41–44) so each delivered node has a `depth`. Give node `id` a depth equal to its id (deterministic, easy to assert):

```python
    def build_routing_tree(self):
        # a node "delivers" iff it is alive and currently reachable;
        # depth = node id (deterministic, for avg_hop assertions)
        return {sid: {'depth': sid} for sid in self._reachable
                if self.sensors[sid].is_alive}
```

`FakeSensor` already has `.power` (line 26), so universal `avg_tx_power` needs no fake change.

- [ ] **Step 2: Write the failing tests for the new metrics**

Add these two test functions to `tests/test_metrics.py` (after `test_pdr_and_energy_per_packet`):

```python
def test_avg_hop_series_and_summary():
    sensors = [FakeSensor(i, e0=10.0) for i in range(4)]
    net = FakeNet(sensors)                      # depth = id -> [0,1,2,3]
    mc = MetricsCollector(net)

    mc.record_round(net, 0, {})
    ts = mc.time_series()
    # avg_hop counts the final hop to BS: mean(depth)+1 = mean([0,1,2,3])+1 = 2.5
    assert abs(ts['avg_hop'][0] - 2.5) < 1e-9, ts['avg_hop']

    net._reachable = set()                      # nobody reaches BS
    mc.record_round(net, 1, {})
    assert ts['avg_hop'][1] == 0.0, ts['avg_hop']   # empty tree -> 0.0

    s = mc.finalize()
    # mean over recorded rounds: (2.5 + 0.0)/2 = 1.25
    assert abs(s['mean_avg_hop'] - 1.25) < 1e-9, s
    print('  avg_hop series/summary OK')


def test_avg_tx_power_universal():
    sensors = [FakeSensor(i, e0=10.0) for i in range(3)]
    for s in sensors:
        s.power = 2.0e-4
    net = FakeNet(sensors)
    mc = MetricsCollector(net)

    mc.record_round(net, 0, {})
    ts = mc.time_series()
    assert abs(ts['avg_tx_power'][0] - 2.0e-4) < 1e-12, ts['avg_tx_power']

    s = mc.finalize()
    assert abs(s['mean_avg_tx_power'] - 2.0e-4) < 1e-12, s
    print('  universal avg_tx_power OK')
```

Register them in the `__main__` block at the bottom (after `test_pdr_and_energy_per_packet()`):

```python
    test_avg_hop_series_and_summary()
    test_avg_tx_power_universal()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `conda run -n base python tests/test_metrics.py`
Expected: FAIL — `KeyError: 'avg_hop'` (or `'avg_tx_power'`) from `time_series()`, since those series do not exist yet.

- [ ] **Step 4: Add the two series to `MetricsCollector.__init__`**

In `metrics.py`, in `__init__` after `self.generated = []` (line ~56), add:

```python
        self.avg_hop = []
        self.avg_tx_power = []
```

- [ ] **Step 5: Compute the new metrics in `record_round` (build the tree once)**

In `metrics.py`, replace the body of `record_round` so the routing tree is built a single time and used for both `delivered` and `avg_hop`, and `avg_tx_power` is computed over alive nodes. Replace the existing assignments (the `delivered = len(net.build_routing_tree())` line and the appends) with:

```python
        alive_sensors = [s for s in net.sensors if s.is_alive]
        alive = len(alive_sensors)
        energies = [s.e_res for s in alive_sensors]
        total_e = float(sum(energies))
        e_std = float(np.std(energies)) if alive >= 2 else 0.0

        tree = net.build_routing_tree()
        delivered = len(tree)
        # avg_hop: mean depth over delivered nodes, +1 for the final hop to the
        # BS (a gateway has depth 0 = one hop to the sink). 0.0 if nothing
        # reaches the BS this round.
        avg_hop = (float(np.mean([n['depth'] for n in tree.values()])) + 1.0
                   if tree else 0.0)
        # avg_tx_power: universal-core mean transmit power over alive nodes.
        avg_tx_power = (float(np.mean([s.power for s in alive_sensors]))
                        if alive_sensors else 0.0)

        self.rounds.append(t)
        self.alive.append(alive)
        self.total_energy.append(total_e)
        self.energy_std.append(e_std)
        self.delivered.append(delivered)
        self.generated.append(alive)
        self.avg_hop.append(avg_hop)
        self.avg_tx_power.append(avg_tx_power)
```

Leave the `if family_extras:` block at the end of `record_round` unchanged.

- [ ] **Step 6: Add the new scalar summaries to `finalize`**

In `metrics.py` `finalize`, after `mean_energy_std = ...` (line ~111) add:

```python
        mean_avg_hop = float(np.mean(self.avg_hop)) if self.avg_hop else 0.0
        mean_avg_tx_power = (float(np.mean(self.avg_tx_power))
                             if self.avg_tx_power else 0.0)
```

Then add both keys to the `self._summary = {...}` dict (after `'mean_energy_std'`):

```python
            'mean_avg_hop': mean_avg_hop,
            'mean_avg_tx_power': mean_avg_tx_power,
```

- [ ] **Step 7: Expose the new series in `time_series`**

In `metrics.py` `time_series`, add the two series to the returned dict (after `'generated': self.generated,`):

```python
            'avg_hop': self.avg_hop,
            'avg_tx_power': self.avg_tx_power,
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `conda run -n base python tests/test_metrics.py`
Expected: PASS — ends with `All metrics tests passed.` (existing tests still pass; the two new lines print `avg_hop series/summary OK` and `universal avg_tx_power OK`).

- [ ] **Step 9: Verify end-to-end on a real short run**

Run:
```bash
conda run -n base python -c "
from main import build_network
from algos.gt2 import GT2
net = build_network(deployment='uniform', num_nodes=40, seed=1)
a = GT2(net, config_path='config/gt2.yaml'); a.max_rounds=3; a.plot_period=10**9; a.run()
ts = a.metrics.time_series(); s = a.metrics.summary()
print('avg_hop[:3]   =', ts['avg_hop'][:3])
print('avg_tx_power  =', ts['avg_tx_power'][:3])
print('mean_avg_hop  =', s['mean_avg_hop'], 'mean_avg_tx_power =', s['mean_avg_tx_power'])
assert 'avg_hop' in ts and 'avg_tx_power' in ts
assert ts['avg_hop'][0] >= 1.0          # GT2 (clustering) reports hop count
assert ts['avg_tx_power'][0] >= 0.0     # clustering algo now reports tx power
print('OK: clustering algo reports both new metrics')
"
```
Expected: prints non-empty series and `OK: clustering algo reports both new metrics`. This confirms a *clustering* algorithm (GT2) now reports `avg_tx_power`, closing the topology-only gap. (No commit — leave changes in the working tree.)

---

## Task 2: Surface the new scalar summaries in `summary.csv`

**Files:**
- Modify: `benchmark.py` (`SUMMARY_FIELDS`, line ~43)

- [ ] **Step 1: Add the new fields to `SUMMARY_FIELDS`**

In `benchmark.py`, extend `SUMMARY_FIELDS` so the per-run CSV carries the new scalars:

```python
SUMMARY_FIELDS = [
    'algo', 'deployment', 'seed', 'num_nodes',
    'fnd', 'hnd', 'lnd',
    'total_delivered', 'total_generated', 'mean_pdr', 'cumulative_pdr',
    'energy_drained', 'energy_per_packet', 'mean_energy_std',
    'mean_avg_hop', 'mean_avg_tx_power',
]
```

- [ ] **Step 2: Verify the CSV builder accepts the new fields**

Run:
```bash
conda run -n base python -c "
from benchmark import SUMMARY_FIELDS
assert 'mean_avg_hop' in SUMMARY_FIELDS and 'mean_avg_tx_power' in SUMMARY_FIELDS
print('SUMMARY_FIELDS OK:', SUMMARY_FIELDS[-2:])
"
```
Expected: `SUMMARY_FIELDS OK: ['mean_avg_hop', 'mean_avg_tx_power']`. (`build_summary_csv` uses `DictWriter(..., extrasaction='ignore')`, so the new keys from each run's summary are written wherever the field is listed.) (No commit.)

---

## Task 3: Per-scenario figures + median energy-per-packet in `plot_benchmark.py`

**Files:**
- Modify: `plot_benchmark.py` (`SCALAR_METRICS`, add `_family_curve`, `generate_all`, `write_scenario_summary`)
- Test: `tests/test_plot_smoke.py`

- [ ] **Step 1: Read the existing plot smoke test to match its pattern**

Run: `conda run -n base python tests/test_plot_smoke.py`
Expected: PASS (establishes the baseline before changes). If the file builds a tiny fake `runs` list and calls `generate_all`/helpers against a temp dir, the new assertions in Step 8 follow that same shape.

- [ ] **Step 2: Add the new per-scenario scalar metrics to `SCALAR_METRICS`**

In `plot_benchmark.py`, extend `SCALAR_METRICS` (line ~25) so the new scalars get by-scenario bar charts and CSV columns:

```python
SCALAR_METRICS = [
    ('fnd', 'First Node Death (round)'),
    ('hnd', 'Half Node Death (round)'),
    ('lnd', 'Last Node Death (round)'),
    ('total_delivered', 'Total packets to BS'),
    ('cumulative_pdr', 'Cumulative PDR'),
    ('energy_per_packet', 'Energy per delivered packet (J)'),
    ('mean_avg_hop', 'Average hop count to BS'),
    ('mean_avg_tx_power', 'Average transmit power per node'),
]
```

- [ ] **Step 3: Add a `_family_curve` helper for family-nested series**

In `plot_benchmark.py`, add this helper after `_curve` (after line ~72). It mirrors `_curve` but reads from `time_series['family'][family_key]` and skips runs that lack the key (so topology algos are absent from CH-count plots, which is correct):

```python
def _family_curve(runs, deployment, family_key, ylabel, fname, fig_dir):
    by_algo = defaultdict(list)
    for r in runs:
        if r['deployment'] != deployment:
            continue
        series = r['time_series'].get('family', {}).get(family_key)
        if series:
            by_algo[r['algo']].append(series)
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
```

- [ ] **Step 4: Emit the new per-scenario over-time figures in `generate_all`**

In `plot_benchmark.py` `generate_all`, inside the existing
`for deployment in sorted({r['deployment'] for r in runs}):` loop (after the two
existing `_curve(...)` calls), add:

```python
        _curve(runs, deployment, 'avg_hop', 'Average hop count to BS',
               f'hop_{deployment}.png', fig_dir)
        _curve(runs, deployment, 'avg_tx_power',
               'Average transmit power per node',
               f'tx_power_{deployment}.png', fig_dir)
        _family_curve(runs, deployment, 'ch_count', 'Cluster-head count',
                      f'ch_count_{deployment}.png', fig_dir)
```

- [ ] **Step 5: Add a median energy-per-packet column to the scenario summary**

In `plot_benchmark.py` `write_scenario_summary`, add a median column for
`energy_per_packet` so the gaussian outlier's effect is explicit. After the
`for key, _ in SCALAR_METRICS: fields += [...]` block (line ~152), append:

```python
    fields += ['energy_per_packet_median']
```

Then in the row-writing loop, after the `for key, _ in SCALAR_METRICS:` inner
loop that sets `row[f'{key}_mean']` / `_std` (line ~165), add:

```python
            epp = c['energy_per_packet']
            row['energy_per_packet_median'] = (float(np.median(epp))
                                               if epp else '')
```

- [ ] **Step 6: Regenerate figures against the EXISTING (old-schema) runs as a smoke check**

The current `results/runs/*.json` lack `avg_hop`/`avg_tx_power` as core series, but topology runs still have `avg_degree`/`ch_count` family data, and `_curve` for missing core keys must not crash. Guard `_curve` against a missing `ts_key`: in `plot_benchmark.py` `_curve`, change the append line

```python
        by_algo[r['algo']].append(r['time_series'][ts_key])
```
to
```python
        series = r['time_series'].get(ts_key)
        if series is not None:
            by_algo[r['algo']].append(series)
```

This makes the new `hop_*`/`tx_power_*` curves simply skip old-schema runs instead of raising `KeyError`.

- [ ] **Step 7: Add smoke-test assertions for the new code paths**

In `tests/test_plot_smoke.py`, extend the fake run(s) so each has
`time_series` containing `avg_hop`, `avg_tx_power`, and a `family` dict with
`ch_count`, plus `summary` containing `mean_avg_hop`, `mean_avg_tx_power`,
`energy_per_packet`. Then assert the new outputs exist. Add (adapting variable
names to the file's existing fixture):

```python
    # new per-scenario figures exist
    import os
    for dep in deployments:                      # deployments from the fixture
        assert os.path.exists(os.path.join(fig_dir, f'hop_{dep}.png'))
        assert os.path.exists(os.path.join(fig_dir, f'tx_power_{dep}.png'))
    # median epp column present in the scenario summary
    with open(os.path.join(tmp, 'summary_by_scenario.csv')) as f:
        header = f.readline()
    assert 'energy_per_packet_median' in header
    print('  new figures + median column OK')
```

If `tests/test_plot_smoke.py` does not already build per-`time_series` fakes, add a minimal fake run dict in the fixture:

```python
    def fake_run(algo, dep, seed):
        return {
            'algo': algo, 'deployment': dep, 'seed': seed, 'num_nodes': 4,
            'summary': {'fnd': 5, 'hnd': 8, 'lnd': 10, 'total_delivered': 100,
                        'total_generated': 200, 'mean_pdr': 0.5,
                        'cumulative_pdr': 0.5, 'energy_drained': 1.0,
                        'energy_per_packet': 0.01, 'mean_energy_std': 0.1,
                        'mean_avg_hop': 2.0, 'mean_avg_tx_power': 1e-4},
            'time_series': {
                'rounds': [0, 1], 'alive': [4, 3], 'total_energy': [40.0, 36.0],
                'energy_std': [0.0, 0.1], 'delivered': [4, 3],
                'generated': [4, 4], 'avg_hop': [2.0, 2.5],
                'avg_tx_power': [1e-4, 1e-4],
                'family': {'ch_count': [1, 1], 'cluster_sizes': [[2], [2]]}},
        }
```

- [ ] **Step 8: Run the plot smoke test**

Run: `conda run -n base python tests/test_plot_smoke.py`
Expected: PASS — prints `new figures + median column OK` among the existing output. (No commit.)

---

## Task 4: Back up prior results and re-run the sweep

**Files:** none edited — operational task producing `results/runs/*.json`, `results/summary.csv`, `results/summary_by_scenario.csv`, `results/figures/*.png`.

- [ ] **Step 1: Back up the prior run (schema changed — old JSONs must be regenerated)**

```bash
cd /home/quanh/Documents/workspace/github/gt-tc
mv results/runs results/runs_backup
cp results/summary.csv results/summary.csv.bak
cp results/summary_by_scenario.csv results/summary_by_scenario.csv.bak
mkdir -p results/runs
```
Expected: `results/runs_backup/` holds the old JSONs; `results/runs/` is empty so the resumable sweep regenerates everything with the new schema.

- [ ] **Step 2: Run the sweep**

Run: `conda run -n base python benchmark.py`
Expected: `Dispatching 660 runs across <N> workers...`, then per-run completion lines, ending with the `summary.csv` path. (Resumable: re-running skips finished JSONs.) This is the long step — let it finish.

- [ ] **Step 3: Verify every regenerated JSON carries the new core series**

```bash
conda run -n base python -c "
import glob, json
bad = []
for f in glob.glob('results/runs/*.json'):
    ts = json.load(open(f))['time_series']
    if 'avg_hop' not in ts or 'avg_tx_power' not in ts:
        bad.append(f)
print('total runs:', len(glob.glob('results/runs/*.json')))
print('missing new series:', len(bad))
assert not bad, bad[:5]
# clustering algo must now have avg_tx_power populated
d = json.load(open(glob.glob('results/runs/GT2_uniform_n200_1.json')[0]))
assert d['time_series']['avg_tx_power'][0] >= 0.0
print('OK: all runs carry avg_hop + avg_tx_power; clustering reports tx power')
"
```
Expected: `missing new series: 0` and `OK: ...`.

- [ ] **Step 4: Regenerate figures and summaries**

Run: `conda run -n base python plot_benchmark.py`
Expected: `Figures written to results/figures` and `Per-scenario summary written to results/summary_by_scenario.csv`.

- [ ] **Step 5: Verify the new figures and CSV columns exist**

```bash
conda run -n base python -c "
import glob, os
figs = os.listdir('results/figures')
for dep in ['uniform_n200','gaussian_n100','cov_free_n40_r90']:
    for pre in ['hop_','tx_power_','ch_count_','survival_']:
        assert f'{pre}{dep}.png' in figs, f'{pre}{dep}.png missing'
hdr = open('results/summary_by_scenario.csv').readline()
for col in ['mean_avg_hop_mean','mean_avg_tx_power_mean','energy_per_packet_median']:
    assert col in hdr, col
print('OK: new figures + scenario-summary columns present')
"
```
Expected: `OK: new figures + scenario-summary columns present`.

- [ ] **Step 6: Sanity-check consistency with the backed-up run (lifetime unchanged)**

```bash
conda run -n base python -c "
import csv
def load(p):
    return {(r['deployment'], r['algo']): r for r in csv.DictReader(open(p))}
new = load('results/summary_by_scenario.csv')
old = load('results/summary_by_scenario.csv.bak')
key = ('uniform_n200','GT2')
print('GT2 uniform_n200 FND  new:', new[key]['fnd_mean'], ' old:', old[key]['fnd_mean'])
# FND is deterministic from (deployment,num_nodes,seed); should match closely
assert abs(float(new[key]['fnd_mean']) - float(old[key]['fnd_mean'])) < 1e-6
print('OK: lifetime metrics reproduced exactly')
"
```
Expected: matching FND values and `OK: lifetime metrics reproduced exactly`. (If they differ, the re-run is not reproducing the prior sweep — investigate before proceeding, do not overwrite the backup.) The regenerated artifacts stay in the working tree — **no commit**.

---

## Task 5: Write the §5.2–5.4 guideline into the paper's `.tex` file

**Files:**
- Modify: `~/Documents/workspace/github/GT2_paper/sec4_simulation.tex` (the three empty subsections at lines ~44–51)

- [ ] **Step 1: Pull the concrete numbers for each subsection**

```bash
conda run -n base python -c "
import csv
rows=list(csv.DictReader(open('results/summary_by_scenario.csv')))
cols=['fnd_mean','hnd_mean','lnd_mean','total_delivered_mean','cumulative_pdr_mean',
      'energy_per_packet_mean','energy_per_packet_median','mean_avg_hop_mean','mean_avg_tx_power_mean']
for scen in sorted(set(r['deployment'] for r in rows)):
    print('===', scen, '===')
    for r in sorted([x for x in rows if x['deployment']==scen], key=lambda x: -float(x['fnd_mean'] or 0)):
        vals=' '.join(f'{c.replace(\"_mean\",\"\").replace(\"_\",\"\"):>14}={r[c]}' for c in cols)
        print(f\"{r['algo']:13} \"+vals)
" > /tmp/sec5_numbers.txt
echo "wrote /tmp/sec5_numbers.txt"; head -30 /tmp/sec5_numbers.txt
```
Expected: a per-scenario table of every metric the guideline will cite. Keep this open while writing Step 2.

- [ ] **Step 2: Read the target subsections in the paper**

Read `~/Documents/workspace/github/GT2_paper/sec4_simulation.tex` lines ~44–51
(the three empty subsections `\subsection{Network lifetime}`,
`\subsection{Network behaviour}`, `\subsection{Network performance}`). The
guideline bullets get inserted *under each* `\subsection{...}\label{...}` line,
as a LaTeX comment block (lines starting with `%`) so they do not render — the
user writes prose around them.

- [ ] **Step 3: Insert the guideline comment block under §5.2 (Network lifetime)**

Using Edit, insert after the `\label{subsec: lifetime_bench}` line. Fill every
`<…>` with real numbers from `/tmp/sec5_numbers.txt` (do **not** leave
placeholders — e.g. "GT2 FND ≈ 13049 vs LEACH 7519, 1.7×"). G = GT2-vs-baseline
lens, N/E = normal-vs-extreme lens.

```latex
% WRITING GUIDELINE (delete when written). Normal=uniform_n200 (dense, even);
% extreme=gaussian_n100 (clustered) + cov_*_n40/n60 (sparse, +/- obstacle).
% Figures: figures/fnd_by_scenario.png, hnd_by_scenario.png, lnd_by_scenario.png,
%          figures/survival_<scenario>.png
% - [G] FND ranking, normal case: <GT2 vs top baselines on uniform_n200, numbers>.
% - [G] HND/LND and graceful degradation: <numbers + survival-curve shape>.
% - [N/E] How GT2's FND lead shifts normal -> sparse -> clustered: <numbers>.
% - [N/E] Which baselines collapse earliest under extreme deployments: <numbers>.
```

- [ ] **Step 4: Insert the guideline comment block under §5.3 (Network behaviour)**

Insert after `\label{subsec: behaviour_bench}`, numbers filled in:

```latex
% WRITING GUIDELINE (delete when written).
% Figures: figures/hop_<scenario>.png, figures/ch_count_<scenario>.png
% - [G] Avg hop-to-BS over time, normal case, GT2 vs baselines: <numbers/trend>.
% - [G] CH-count trajectory: how GT2's clustering game sizes the CH population: <numbers>.
% - [N/E] Hop count growth sparse vs clustered (longer multi-hop paths): <numbers>.
% - [N/E] CH-count behaviour as nodes die, normal vs extreme: <trend>.
```

- [ ] **Step 5: Insert the guideline comment block under §5.4 (Network performance)**

Insert after `\label{subsec: performance_bench}`, numbers filled in:

```latex
% WRITING GUIDELINE (delete when written).
% Figures: figures/tx_power_<scenario>.png, tx_power_by_scenario.png,
%          total_delivered_by_scenario.png, energy_per_packet_by_scenario.png
% - [G] Avg transmit power per node, GT2 vs baselines: <numbers>.
% - [G] Throughput (packets to BS) and cumulative PDR: <numbers>.
% - [G] Energy-per-packet by scenario, GT2 standing: <numbers>.
% - [PER-ALGO] EFTCG: best energy-per-packet in 5/6 scenarios (<numbers>), but on
%   gaussian_n100 mean epp blows up to <mean> while median is <median> -- caused by
%   sink-disconnection (k-connectivity utility ignores the BS; first power-reduction
%   step severs the long links reaching the central sink in clustered layouts).
%   Contrast GT2's BS-rooted clustering, which never partitions the sink.
%   One observation among several -- also note <one line each: DIA/TCLE/MIA etc>.
% - [N/E] How transmit power / throughput shift normal -> extreme: <numbers>.
% TABLES to add under table/: (1) lifetime FND/HND/LND per (algo,scenario);
%   (2) performance: energy-per-packet mean AND median, avg tx power, throughput.
```

- [ ] **Step 6: Verify no placeholders remain in the inserted blocks**

```bash
grep -n '<' ~/Documents/workspace/github/GT2_paper/sec4_simulation.tex && echo "PLACEHOLDERS REMAIN — fill them" || echo "OK: no placeholders"
```
Expected: `OK: no placeholders`. If any `<…>` remain, fill them from `/tmp/sec5_numbers.txt`. (No commit — the paper repo is the user's to commit.)

---

## Self-Review notes

- **Spec coverage:** Part A → Tasks 1–2 + Task 4; Part B → Task 3 + Task 4; Part C → Task 5 (guideline written as LaTeX comments directly into `sec4_simulation.tex`). All three acceptance criteria for data/figures are verified in Task 4 Steps 3/5/6; the guideline (criterion 4) is Task 5.
- **No commits:** every task leaves changes in the working tree; the user commits.
- **Type consistency:** new keys `avg_hop`, `avg_tx_power` (series) and `mean_avg_hop`, `mean_avg_tx_power` (summary scalars) are used identically across `metrics.py`, `benchmark.py` (`SUMMARY_FIELDS`), and `plot_benchmark.py` (`SCALAR_METRICS`). The figure naming `hop_/tx_power_/ch_count_<deployment>.png` is consistent between Task 3 (creation) and Task 4/Task 5 (verification/citation).
- **avg_hop semantics:** defined as `mean(depth)+1` so a direct-to-BS gateway counts as 1 hop, matching "shortest path to BS" in the paper; empty tree → 0.0.
```
