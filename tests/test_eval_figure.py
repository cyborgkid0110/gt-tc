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
