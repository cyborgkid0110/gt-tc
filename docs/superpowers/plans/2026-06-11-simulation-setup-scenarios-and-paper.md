# Simulation-Setup: Scenarios + Paper Subsection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the six evaluation deployment scenarios (4 new coverage-PSO configs × 10 seeds), render a single 2×3 deployment figure, and write the "Simulation setup" subsection (with two parameter tables) of the paper.

**Architecture:** Extend the existing `scenarios/` geometry+placement pipeline with a circular-obstacle option and a parallel batch driver; add a standalone figure renderer; append LaTeX to the separate `GT2_paper` repo. The energy/coverage machinery is reused unchanged — only a `circles` geometry key and a batch harness are added.

**Tech Stack:** Python (numpy, shapely, matplotlib, pyyaml), Anaconda `base` env, LaTeX (`DCV_template.cls`, `latexmk`).

**Conventions (read before starting):**
- Run everything in the `base` conda env: prefix commands with `conda run -n base`.
- There is **no pytest** in the env. Tests are plain scripts run with `conda run -n base python tests/<file>.py`; each test file ends with an `if __name__ == '__main__':` block that calls every test function. When you add a test, also add its call to that block.
- Branch first (don't work on `main`): in the gt-tc repo create `git switch -c feat/sim-setup-scenarios`. The paper is a **separate git repo** at `~/Documents/workspace/github/GT2_paper`; create a branch there too (`git -C ~/Documents/workspace/github/GT2_paper switch -c feat/simulation-setup`).
- Two repos: gt-tc tasks commit in gt-tc; paper tasks (Task 6) commit in the GT2_paper repo.

**Scenario file inventory (target):** all on ±250 m area, seeds `{1,2,3,4,5,6,7,8,9,42}`.

| Config tag | Nodes | cov_radius | Obstacle | BS | Status |
|------------|-------|-----------|----------|----|--------|
| `uniform_n200` | 200 | — | none | (0,0) | exists |
| `gaussian_n100` | 100 | — | none | (0,0) | exists |
| `cov_free_n60_r60` | 60 | 60 | none | (0,0) | new |
| `cov_free_n40_r90` | 40 | 90 | none | (0,0) | new |
| `cov_obs_n60_r60` | 60 | 60 | circle r=80 @ (0,0) | (180,−180) | new |
| `cov_obs_n40_r90` | 40 | 90 | circle r=80 @ (0,0) | (180,−180) | new |

---

## File Structure

- **Modify** `scenarios/regions.py` — accept a `circles` key in `Region.from_def`.
- **Create** `scenarios/freeze_coverage_scenarios.py` — parallel, idempotent batch driver for the 4 coverage configs × 10 seeds.
- **Modify** `plot_deployments.py` — draw `circles` in the `--def` render path.
- **Create** `plot_eval_scenarios.py` — the combined 2×3 deployment figure.
- **Modify** `tests/test_regions.py` — circle-obstacle test.
- **Create** `tests/test_freeze_coverage.py` — single-instance generation test (fast PSO config).
- **Create** `tests/test_eval_figure.py` — smoke test for the 2×3 figure.
- **Create** `scenarios/gen/cov_*_s*.csv` — 40 generated CSVs (Task 3 output).
- **Modify** `~/.../GT2_paper/sec4_simulation.tex` — add the subsection + figure.
- **Create** `~/.../GT2_paper/table/simulation_parameters.tex`, `~/.../GT2_paper/table/algorithm_parameters.tex`.
- **Create** `~/.../GT2_paper/figures/deployment_scenarios.png` (Task 5 output, copied from repo).

---

## Task 1: Circular obstacle support in `Region`

**Files:**
- Modify: `scenarios/regions.py` (the `from_def` classmethod, around lines 22–41)
- Test: `tests/test_regions.py`

- [ ] **Step 1: Write the failing test**

Add this function to `tests/test_regions.py` immediately before the `if __name__ == '__main__':` block:

```python
def test_circle_obstacle_excluded():
    d = {'area': 250, 'num_nodes': 10, 'coverage_radius': 40, 'seed': 0,
         'bs': [180.0, -180.0], 'obstacles': [],
         'circles': [[0.0, 0.0, 80.0]], 'paths': []}
    r = Region.from_def(d)
    assert not r.contains((0.0, 0.0)), "centre is inside the circle obstacle"
    assert not r.contains((50.0, 0.0)), "inside the circle radius"
    assert r.contains((180.0, -180.0)), "BS location must be deployable"
    print("  circle obstacle exclusion OK")
```

And add its call inside the `__main__` block (after `test_coverage_targets_in_region()`):

```python
    test_circle_obstacle_excluded()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n base python tests/test_regions.py`
Expected: FAILS — the BS at (180,−180) is currently fine, but circles are ignored so `r.contains((0.0, 0.0))` returns True and the first assert raises `AssertionError: centre is inside the circle obstacle`.

- [ ] **Step 3: Implement circle support**

In `scenarios/regions.py`, inside `from_def`, replace:

```python
        obstacles = [Polygon(ring) for ring in d.get('obstacles', [])]
        obstacle_union = unary_union(obstacles) if obstacles else None
```

with:

```python
        obstacles = [Polygon(ring) for ring in d.get('obstacles', [])]
        obstacles += [Point(float(cx), float(cy)).buffer(float(rad))
                      for cx, cy, rad in d.get('circles', [])]
        obstacle_union = unary_union(obstacles) if obstacles else None
```

(`Point` is already imported at the top of `regions.py`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n base python tests/test_regions.py`
Expected: PASS — final line `All region tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scenarios/regions.py tests/test_regions.py
git commit -m "feat: circular obstacle support in Region.from_def"
```

---

## Task 2: Batch driver for the coverage scenarios

**Files:**
- Create: `scenarios/freeze_coverage_scenarios.py`
- Test: `tests/test_freeze_coverage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_coverage.py`:

```python
# tests/test_freeze_coverage.py
"""Single-instance coverage generation: in-region, obstacle-free, BS-connected.

Run:  conda run -n base python tests/test_freeze_coverage.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import load_scenario
from scenarios.coverage_deploy import is_connected_to_bs
from scenarios.freeze_coverage_scenarios import freeze_one_coverage

# Fast PSO config so the test does not depend on the tuned config/coverage.yaml.
_FAST_CFG = """
pso_particles: 20
pso_iters: 40
pso_w: 0.7
pso_c1: 1.5
pso_c2: 1.5
penalty_lambda: 2.0
target_spacing: 10
"""


def test_freeze_one_coverage_with_obstacle():
    config = {'tag': 'cov_obs_test', 'num_nodes': 12, 'coverage_radius': 40,
              'circles': [[0.0, 0.0, 30.0]], 'bs': [80.0, -80.0], 'area': 100}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, 'coverage.yaml')
        with open(cfg, 'w') as f:
            f.write(_FAST_CFG)
        path, skipped = freeze_one_coverage(config, seed=1,
                                            config_path=cfg, out_dir=tmp)
        assert not skipped, "first generation should not be skipped"
        assert os.path.exists(path), path

        positions, vpre, bs_pos, meta = load_scenario(path)
        assert positions.shape == (12, 2), positions.shape
        # no node lands inside the central circular obstacle
        dist = np.hypot(positions[:, 0], positions[:, 1])
        assert (dist > 30.0 - 1e-6).all(), "a node is inside the obstacle"
        # BS persisted correctly
        assert abs(bs_pos[0] - 80.0) < 1e-9 and abs(bs_pos[1] + 80.0) < 1e-9
        # connected at the coverage radius
        assert is_connected_to_bs(positions, bs_pos, 40.0), "not BS-connected"
        # meta carries the provenance
        assert meta['source'] == 'cov_obs_test'
        assert int(meta['num_nodes']) == 12

        # second call is idempotent (file exists -> skipped)
        _path2, skipped2 = freeze_one_coverage(config, seed=1,
                                               config_path=cfg, out_dir=tmp)
        assert skipped2, "second generation should be skipped"
    print("  freeze_one_coverage (obstacle) OK")


if __name__ == '__main__':
    test_freeze_one_coverage_with_obstacle()
    print("\nAll freeze-coverage tests passed.")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n base python tests/test_freeze_coverage.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scenarios.freeze_coverage_scenarios'`.

- [ ] **Step 3: Implement the batch driver**

Create `scenarios/freeze_coverage_scenarios.py`:

```python
# scenarios/freeze_coverage_scenarios.py
"""Snapshot the coverage-PSO deployment scenarios to scenarios/gen/*.csv.

Four configurations (two free-space, two with a central circular obstacle) are
each generated for 10 seeds. Each instance is an independent projected-PSO
placement, so instances are produced in parallel. Re-running is idempotent
(existing CSVs are skipped).

Run:  conda run -n base python -m scenarios.freeze_coverage_scenarios
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .regions import Region
from .coverage_deploy import place_nodes
from .store import save_scenario

SCENARIO_DIR = os.path.join('scenarios', 'gen')
AREA = 250
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]

# Each config: tag (file prefix), node count, coverage radius (= deployment comm
# range), circular obstacles [[cx, cy, r], ...], base-station position.
COVERAGE_CONFIGS = [
    {'tag': 'cov_free_n60_r60', 'num_nodes': 60, 'coverage_radius': 60,
     'circles': [], 'bs': [0.0, 0.0]},
    {'tag': 'cov_free_n40_r90', 'num_nodes': 40, 'coverage_radius': 90,
     'circles': [], 'bs': [0.0, 0.0]},
    {'tag': 'cov_obs_n60_r60', 'num_nodes': 60, 'coverage_radius': 60,
     'circles': [[0.0, 0.0, 80.0]], 'bs': [180.0, -180.0]},
    {'tag': 'cov_obs_n40_r90', 'num_nodes': 40, 'coverage_radius': 90,
     'circles': [[0.0, 0.0, 80.0]], 'bs': [180.0, -180.0]},
]


def _def_for(config, seed):
    """Build a region-definition dict for one (config, seed) instance."""
    return {
        'name': f"{config['tag']}_s{seed}",
        'area': config.get('area', AREA),
        'num_nodes': config['num_nodes'],
        'coverage_radius': config['coverage_radius'],
        'seed': seed,
        'bs': config['bs'],
        'obstacles': [],
        'circles': config['circles'],
        'paths': [],
    }


def freeze_one_coverage(config, seed, config_path='config/coverage.yaml',
                        out_dir=SCENARIO_DIR):
    """Generate one coverage scenario CSV. Returns (path, skipped)."""
    d = _def_for(config, seed)
    path = os.path.join(out_dir, f"{d['name']}.csv")
    if os.path.exists(path):
        return path, True

    region = Region.from_def(d)
    radius = float(d['coverage_radius'])
    positions = place_nodes(region, d['num_nodes'], radius, seed, config_path)
    rng = np.random.default_rng(seed)
    vpre = rng.uniform(2.7, 4.2, size=len(positions))

    os.makedirs(out_dir, exist_ok=True)
    save_scenario(path, positions, vpre, region.bs_pos,
                  {'source': config['tag'], 'num_nodes': len(positions),
                   'seed': seed, 'area': d['area'], 'coverage_radius': radius})
    return path, False


def main(workers=None):
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    jobs = [(c, s) for c in COVERAGE_CONFIGS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        futs = {ex.submit(freeze_one_coverage, c, s): (c['tag'], s)
                for c, s in jobs}
        for fut in as_completed(futs):
            tag, seed = futs[fut]
            path, skipped = fut.result()
            print(f"{'skip' if skipped else 'wrote'} {path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n base python tests/test_freeze_coverage.py`
Expected: PASS — final line `All freeze-coverage tests passed.` (takes a few seconds: 40 fast PSO iters on 12 nodes.)

- [ ] **Step 5: Commit**

```bash
git add scenarios/freeze_coverage_scenarios.py tests/test_freeze_coverage.py
git commit -m "feat: batch driver for coverage-PSO deployment scenarios"
```

---

## Task 3: Generate the 40 coverage scenarios

**Files:**
- Create (output): `scenarios/gen/cov_free_n60_r60_s*.csv`, `cov_free_n40_r90_s*.csv`, `cov_obs_n60_r60_s*.csv`, `cov_obs_n40_r90_s*.csv` (40 files)

This task runs the real PSO (`config/coverage.yaml`, `pso_iters=3000`) for 40 independent instances in parallel. It is long-running (minutes); run it in the background.

- [ ] **Step 1: Launch the batch generation in the background**

Run (in background): `conda run -n base python -m scenarios.freeze_coverage_scenarios`
Expected: streams `wrote scenarios/gen/cov_*_s*.csv` lines as instances finish; no `RuntimeError` (the connectivity repair guarantees a BS-connected layout). If a `RuntimeError: Could not repair connectivity` appears for any instance, stop and report — it means the node count is too low for that radius/region (should not happen for these configs).

- [ ] **Step 2: Verify all 40 files exist**

Run: `ls scenarios/gen/cov_*_s*.csv | wc -l`
Expected: `40`

- [ ] **Step 3: Verify obstacle scenarios exclude the central circle and place the BS correctly**

Run:
```bash
conda run -n base python -c "
import glob, numpy as np
from scenarios import load_scenario
for f in sorted(glob.glob('scenarios/gen/cov_obs_*_s*.csv')):
    pos, vpre, bs, meta = load_scenario(f)
    d = np.hypot(pos[:,0], pos[:,1])
    assert (d > 80.0 - 1e-6).all(), f'node inside obstacle in {f}'
    assert abs(bs[0]-180.0)<1e-9 and abs(bs[1]+180.0)<1e-9, f'bad BS in {f}'
print('obstacle scenarios OK:', len(glob.glob('scenarios/gen/cov_obs_*_s*.csv')), 'files')
"
```
Expected: `obstacle scenarios OK: 20 files` with no assertion error.

- [ ] **Step 4: Verify every generated scenario is BS-connected at its coverage radius**

Run:
```bash
conda run -n base python -c "
import glob
from scenarios import load_scenario
from scenarios.coverage_deploy import is_connected_to_bs
for f in sorted(glob.glob('scenarios/gen/cov_*_s*.csv')):
    pos, vpre, bs, meta = load_scenario(f)
    r = float(meta['coverage_radius'])
    assert is_connected_to_bs(pos, bs, r), f'{f} not connected at r={r}'
print('all coverage scenarios BS-connected')
"
```
Expected: `all coverage scenarios BS-connected`

- [ ] **Step 5: Commit the generated scenarios**

```bash
git add scenarios/gen/cov_free_n60_r60_s*.csv scenarios/gen/cov_free_n40_r90_s*.csv scenarios/gen/cov_obs_n60_r60_s*.csv scenarios/gen/cov_obs_n40_r90_s*.csv
git commit -m "feat: freeze 40 coverage-PSO deployment scenarios"
```

---

## Task 4: Draw circular obstacles in `plot_deployments.py --def`

**Files:**
- Modify: `plot_deployments.py` (the `plot_from_def` function)
- Test: extend `tests/test_freeze_coverage.py` is not appropriate — create a focused smoke test inline in this task

- [ ] **Step 1: Add the circle-drawing loop**

In `plot_deployments.py`, inside `plot_from_def`, find the obstacle-polygon loop:

```python
    for ring in d.get('obstacles', []):
        xs, ys = zip(*ring)
        ax.fill(xs, ys, facecolor=OBSTACLE_COLOR, alpha=0.55, hatch='xx',
                edgecolor=OBSTACLE_COLOR, lw=1.0, zorder=2)
```

Immediately after it, add:

```python
    for cx, cy, rad in d.get('circles', []):
        ax.add_patch(Circle((cx, cy), rad, facecolor=OBSTACLE_COLOR, alpha=0.55,
                            hatch='xx', edgecolor=OBSTACLE_COLOR, lw=1.0, zorder=2))
```

(`Circle` is already imported at the top of `plot_deployments.py`.)

- [ ] **Step 2: Smoke-check the `--def` renderer with a circle**

This change is exercised end-to-end by the Task 5 figure; here just confirm `plot_from_def` runs with a `circles` def without error. Run:

```bash
conda run -n base python -c "
import os, tempfile, yaml
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from scenarios.store import save_scenario
import plot_deployments as pd

# a tiny frozen CSV at the path plot_from_def expects (scenarios/gen/<name>.csv)
name = '_smoke_circle'
csv = os.path.join('scenarios', 'gen', name + '.csv')
pos = np.array([[10.0, 10.0], [-90.0, 40.0], [120.0, -120.0]])
save_scenario(csv, pos, np.array([3.0, 3.5, 4.0]), (180.0, -180.0),
              {'source': name, 'num_nodes': 3, 'seed': 0, 'area': 250,
               'coverage_radius': 60})
defn = {'name': name, 'area': 250, 'num_nodes': 3, 'coverage_radius': 60,
        'seed': 0, 'bs': [180.0, -180.0], 'obstacles': [],
        'circles': [[0.0, 0.0, 80.0]], 'paths': []}
dpath = os.path.join(tempfile.gettempdir(), name + '.yaml')
with open(dpath, 'w') as f:
    yaml.safe_dump(defn, f)
out = os.path.join(tempfile.gettempdir(), name + '.png')
pd.plot_from_def(dpath, coverage_disks=False, show_links=False, out=out)
assert os.path.exists(out), out
os.remove(csv); os.remove(dpath); os.remove(out)
print('plot_from_def circle smoke OK')
"
```
Expected: `wrote <tmp>/_smoke_circle.png` then `plot_from_def circle smoke OK`.

- [ ] **Step 3: Commit**

```bash
git add plot_deployments.py
git commit -m "feat: render circular obstacles in plot_deployments --def mode"
```

---

## Task 5: Combined 2×3 deployment figure

**Files:**
- Create: `plot_eval_scenarios.py`
- Test: `tests/test_eval_figure.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_figure.py`:

```python
# tests/test_eval_figure.py
"""Smoke test: the 2x3 evaluation-scenario figure renders to a PNG.

Run:  conda run -n base python tests/test_eval_figure.py
"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.store import save_scenario
import plot_eval_scenarios


def _fake_gen(gen_dir):
    """Write the six seed-1 CSVs plot_eval_scenarios expects, with few nodes."""
    specs = [
        ('uniform_n200_s1.csv', (0.0, 0.0)),
        ('gaussian_n100_s1.csv', (0.0, 0.0)),
        ('cov_free_n60_r60_s1.csv', (0.0, 0.0)),
        ('cov_free_n40_r90_s1.csv', (0.0, 0.0)),
        ('cov_obs_n60_r60_s1.csv', (180.0, -180.0)),
        ('cov_obs_n40_r90_s1.csv', (180.0, -180.0)),
    ]
    rng = np.random.default_rng(0)
    for name, bs in specs:
        pos = rng.uniform(-200, 200, size=(8, 2))
        save_scenario(os.path.join(gen_dir, name), pos, rng.uniform(2.7, 4.2, 8),
                      bs, {'source': name, 'num_nodes': 8, 'seed': 1,
                           'area': 250, 'coverage_radius': 60})


def test_eval_figure_renders():
    with tempfile.TemporaryDirectory() as tmp:
        gen = os.path.join(tmp, 'gen'); os.makedirs(gen)
        out = os.path.join(tmp, 'out')
        _fake_gen(gen)
        plot_eval_scenarios.generate(gen_dir=gen, out_dirs=(out,))
        png = os.path.join(out, 'deployment_scenarios.png')
        assert os.path.exists(png), png
    print("  eval figure renders OK")


if __name__ == '__main__':
    test_eval_figure_renders()
    print("\nEval figure smoke test passed.")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n base python tests/test_eval_figure.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'plot_eval_scenarios'`.

- [ ] **Step 3: Implement the figure renderer**

Create `plot_eval_scenarios.py`:

```python
"""Render the six evaluation deployment scenarios as one 2x3 panel figure.

One representative instance (seed 1) of each configuration; free-space panels are
plain scatters, obstacle panels draw the central circle and the off-centre BS.

Run:  conda run -n base python plot_eval_scenarios.py
Output: GT2_paper/figures/deployment_scenarios.png + docs/figures/deployment_scenarios.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scenarios import load_scenario

AREA = 250
GEN = os.path.join('scenarios', 'gen')
PAPER_FIG = os.path.expanduser(
    '~/Documents/workspace/github/GT2_paper/figures')
REPO_FIG = os.path.join('docs', 'figures')
OBSTACLE_COLOR = '#d9534f'

# (csv basename, panel title, [(cx, cy, r), ...] obstacles)
PANELS = [
    ('uniform_n200_s1.csv',     'Free space (N=200)',        []),
    ('gaussian_n100_s1.csv',    'Target region (N=100)',     []),
    ('cov_free_n60_r60_s1.csv', 'Free space (N=60, r=60 m)', []),
    ('cov_free_n40_r90_s1.csv', 'Free space (N=40, r=90 m)', []),
    ('cov_obs_n60_r60_s1.csv',  'Obstacle (N=60, r=60 m)',   [(0.0, 0.0, 80.0)]),
    ('cov_obs_n40_r90_s1.csv',  'Obstacle (N=40, r=90 m)',   [(0.0, 0.0, 80.0)]),
]
LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def _panel(ax, gen_dir, csv, title, circles, label):
    positions, _vpre, bs_pos, _meta = load_scenario(os.path.join(gen_dir, csv))
    for cx, cy, rad in circles:
        ax.add_patch(Circle((cx, cy), rad, facecolor=OBSTACLE_COLOR, alpha=0.5,
                            hatch='xx', edgecolor=OBSTACLE_COLOR, lw=1.0,
                            zorder=2))
    ax.scatter(positions[:, 0], positions[:, 1], s=8, c='#1f77b4',
               edgecolors='none', alpha=0.85, zorder=3)
    ax.scatter([bs_pos[0]], [bs_pos[1]], marker='*', s=160, c='red',
               edgecolors='black', linewidths=0.4, zorder=4)
    ax.set_xlim(-AREA, AREA)
    ax.set_ylim(-AREA, AREA)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.set_title(f'{label} {title}', fontsize=9)


def generate(gen_dir=GEN, out_dirs=(PAPER_FIG, REPO_FIG)):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (csv, title, circles), label in zip(axes.ravel(), PANELS, LABELS):
        _panel(ax, gen_dir, csv, title, circles, label)
    fig.tight_layout()
    written = []
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, 'deployment_scenarios.png')
        fig.savefig(out, dpi=130)
        written.append(out)
        print(f'wrote {out}')
    plt.close(fig)
    return written


if __name__ == '__main__':
    generate()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n base python tests/test_eval_figure.py`
Expected: PASS — final line `Eval figure smoke test passed.`

- [ ] **Step 5: Render the real figure (depends on Task 3 output)**

Run: `conda run -n base python plot_eval_scenarios.py`
Expected: two `wrote ...deployment_scenarios.png` lines (one in the paper repo's `figures/`, one in `docs/figures/`).

- [ ] **Step 6: Commit (gt-tc repo)**

```bash
git add plot_eval_scenarios.py tests/test_eval_figure.py docs/figures/deployment_scenarios.png
git commit -m "feat: combined 2x3 deployment-scenario figure"
```

---

## Task 6: Paper "Simulation setup" subsection + parameter tables

**Files:** (all in the **GT2_paper** repo `~/Documents/workspace/github/GT2_paper`)
- Create: `table/simulation_parameters.tex`
- Create: `table/algorithm_parameters.tex`
- Modify: `sec4_simulation.tex`
- Asset: `figures/deployment_scenarios.png` (already written by Task 5 Step 5)

- [ ] **Step 1: Create the shared-parameters table**

Create `~/Documents/workspace/github/GT2_paper/table/simulation_parameters.tex`:

```latex
\begin{table}
\centering
\caption{Shared simulation, radio, and energy-model parameters.}
\label{tab:sim_params}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{cc}
\hline
\hline
Parameter & Value \\
\hline
Field area & $500 \times 500$ m$^2$ \\
Number of nodes $N$ & 40 / 60 / 100 / 200 \\
Initial energy $E_0$ & 5 mJ \\
Min.\ transmit power $p_{\min}$ & $3.0\times10^{-5}$ W \\
Max.\ transmit power $p_{\max}$ & $2.5\times10^{-4}$ W \\
Power step $p_{\mathrm{step}}$ & $3.0\times10^{-6}$ W \\
Max.\ hop count & 3 \\
Required SNR & 10 (10 dB) \\
Receiver noise figure & 6.31 (8 dB) \\
Noise PSD $N_0$ & $3.98\times10^{-21}$ W/Hz \\
Channel bandwidth $B$ & 3 MHz \\
Wavelength $\lambda$ & 0.125 m (2.4 GHz) \\
Path-loss exponent $\gamma$ & 2.0 \\
Antenna gain $G$ & 1.0 (0 dBi) \\
PA efficiency $\eta$ & 0.30 \\
Data rate $R_b$ & 250 kbps \\
Receiver sensitivity $P_{th}$ & $7.53\times10^{-13}$ W \\
Electronics energy $E_{elec}$ & 50 nJ/bit \\
Aggregation energy $E_{agg}$ & 5 nJ/bit \\
Data-packet payload & 32 bits \\
Aggregated-packet payload & 72 bits \\
Sensor sample size & 16 bits \\
Maximum rounds & 50\,000 \\
\hline
\hline
\end{tabular}
\end{table}
```

- [ ] **Step 2: Create the per-algorithm hyperparameter table**

Create `~/Documents/workspace/github/GT2_paper/table/algorithm_parameters.tex`:

```latex
\begin{table}
\centering
\caption{Distinctive hyperparameters of the benchmarked protocols.}
\label{tab:algo_params}
\renewcommand{\arraystretch}{1.3}
\small
\begin{tabular}{l p{0.66\linewidth}}
\hline
\hline
Protocol & Parameters \\
\hline
GT2 & $\rho = 3.4\times10^{-5}$, $\alpha = 1.5$, $\beta = 0.1$, $\mu = 0.01$ \\
LEACH & $p_{ch} = 0.05$ \\
GTFR & FCM cluster fraction $0.05$, fuzziness $m = 2$, penalty weights $\psi = 0.25$ each \\
DIA / MIA & better-response mode (restrained / greedy); LDIA hop limit $k$ (disabled by default) \\
TCLE & $\varepsilon = 0.01$, quadratic pricing, $\mu = 0.01$, $K = 10$ levels, $\tau = 1.0$, $\sigma_{\max} = 0.1$ \\
EFTCG-1 / EFTCG-2 & connectivity order $k = 1$ / $k = 2$ \\
FL-LEACH-PSO & 30 particles, $w = 0.7$, $c_1 = c_2 = 1.5$, gap statistic $k_{\max} = 20$ \\
SCA-L\'evy & $m = 30$ groupings, $T = 50$ iterations, $a = 2.0$, $b = 0.5$, L\'evy $\beta = 1.5$, $p_{ch} = 0.05$ \\
FC-CRA & $P = 0.05$, $d_{\max}$ factor $0.5$, reallocation threshold $0.5$ \\
\hline
\hline
\end{tabular}
\end{table}
```

- [ ] **Step 3: Write the subsection**

Replace the entire contents of `~/Documents/workspace/github/GT2_paper/sec4_simulation.tex` with:

```latex
\section{Results and discussion}
\label{sec: simulations}

\subsection{Simulation setup}
\label{subsec: setup}

We evaluate the proposed GT2 protocol against ten representative clustering and
topology-control baselines drawn from the recent wireless-sensor-network
literature. GT2 is a two-stage game-theoretic protocol that couples a
mixed-strategy clustering game for cluster-head election with a pure-strategy
power-control game that drives the intra-cluster transmission powers toward a
Nash equilibrium.

The first group of baselines comprises \emph{clustering protocols}. LEACH
performs randomised cluster-head rotation with single-hop intra-cluster
communication. GTFR combines fuzzy $c$-means clustering with a game-theoretic
head selection. FL-LEACH-PSO is a fuzzy-logic LEACH variant whose clusters are
formed once by a hybrid PSO/$k$-means optimiser and whose heads are chosen by a
two-tier Mamdani fuzzy controller. SCA-L\'evy elects heads centrally with a
sine-cosine optimiser augmented by L\'evy mutation. FC-CRA uses an
adaptive-radius clustering rule whose cluster size shrinks near the base station
and as the network ages, with multi-hop forwarding between heads.

The second group comprises \emph{topology-control games} that adjust the
transmission power of each node without forming clusters. DIA and MIA are the
restrained (fair) and greedy variants of an ordinal potential game over node
power. TCLE is an energy-aware game whose benefit term is the algebraic
connectivity of the resulting topology. EFTCG-1 and EFTCG-2 are
energy-efficient and fault-tolerant games that enforce one-connectivity and
two-connectivity, respectively, with self-adaptive energy weights.

We assess the protocols on three families of deployment scenarios, illustrated
in Fig.~\ref{fig:deployment_scenarios}. In the \emph{free-space} scenarios,
nodes are deployed across an open square field with the base station at the
centre. In the \emph{target-region} scenario, sensing demand is concentrated
around several hotspots, so nodes cluster around those regions. In the
\emph{obstacle} scenarios, a circular forbidden region at the centre of the
field blocks deployment and routing, and the base station is displaced toward
the lower-right corner, creating an asymmetric, detour-heavy topology. For each
family we generate ten independent random instances and report every metric as
the average over those instances.

The parameters shared by all protocols are listed in
Table~\ref{tab:sim_params}: the field geometry, the per-node energy budget, the
transmit-power bounds, the link-budget radio model, and the energy-consumption
and packet-structure constants. The distinctive hyperparameters of each
protocol are summarised in Table~\ref{tab:algo_params}.

\input{table/simulation_parameters}
\input{table/algorithm_parameters}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/deployment_scenarios.png}
\caption{Representative instances of the six deployment configurations grouped
into three scenario families: free space (a, c, d), target region (b), and with
a central circular obstacle (e, f). Blue dots are sensor nodes, the red star is
the base station, and the hatched disc is the forbidden region.}
\label{fig:deployment_scenarios}
\end{figure}
```

- [ ] **Step 4: Compile the paper to verify it builds**

Run:
```bash
cd ~/Documents/workspace/github/GT2_paper && conda run -n base latexmk -pdf -interaction=nonstopmode main.tex
```
Expected: `main.pdf` is (re)generated with no fatal errors. Warnings about undefined references/citations are acceptable (the user adds citations later). If `latexmk` is unavailable, use `pdflatex -interaction=nonstopmode main.tex` (run twice for refs). Confirm `main.pdf` exists and is newer than before.

- [ ] **Step 5: Confirm the figure and tables appear**

Open `main.pdf` (or check the log) and confirm: the "Simulation setup" subsection text is present, Fig.~\ref{fig:deployment_scenarios} shows the six labelled panels with the obstacle visible in (e) and (f), and both parameter tables typeset within the column width (no overfull-hbox of the algorithm table — the `p{0.66\linewidth}` column should wrap long rows). If the algorithm table still overflows, reduce `\small` to `\footnotesize` in `table/algorithm_parameters.tex`.

- [ ] **Step 6: Commit (GT2_paper repo)**

```bash
git -C ~/Documents/workspace/github/GT2_paper add sec4_simulation.tex table/simulation_parameters.tex table/algorithm_parameters.tex figures/deployment_scenarios.png
git -C ~/Documents/workspace/github/GT2_paper commit -m "Add Simulation setup subsection with deployment figure and parameter tables"
```

---

## Self-Review

**Spec coverage:**
- Spec change 1 (circle obstacle in `regions.py`) → Task 1. ✓
- Spec change 2 (batch driver) → Task 2 (code) + Task 3 (run). ✓
- Spec change 3 (`plot_deployments.py` circles) → Task 4. ✓
- Spec change 4 (combined 2×3 figure) → Task 5. ✓
- Spec change 5 (paper subsection + two tables, no `\cite{}`) → Task 6. ✓
- Scenario inventory (40 new CSVs, seeds, naming) → Task 3 produces `cov_*_s{1..9,42}.csv`. ✓
- Verification items (40 files, obstacle exclusion, BS position, BS-connectivity, figure, compile) → Task 3 Steps 2–4, Task 5, Task 6 Steps 4–5. ✓
- Out-of-scope (no sweep, no `refs.bib`, no `\cite{}`) → respected; Task 6 adds no citations. ✓

**Placeholder scan:** No TBD/TODO; every code and LaTeX step contains full content. ✓

**Type/name consistency:** `freeze_one_coverage(config, seed, config_path=, out_dir=)` defined in Task 2 and called identically in the Task 2 test and Task 3 worker. `generate(gen_dir=, out_dirs=)` defined in Task 5 and called identically in its test. `circles` key spelled consistently across `regions.py`, `plot_deployments.py`, `plot_eval_scenarios.py`, and the config dicts. Table labels `tab:sim_params` / `tab:algo_params` and figure label `fig:deployment_scenarios` are defined once and referenced in the prose. ✓
