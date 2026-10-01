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

from plot_style import DPI, apply_paper_style
from scenarios import load_scenario

apply_paper_style()

AREA = 250
GEN = os.path.join('scenarios', 'gen')
PAPER_FIG = os.path.expanduser(
    '~/Documents/workspace/github/GT2_paper/figures')
REPO_FIG = os.path.join('docs', 'figures')
OBSTACLE_COLOR = '#d9534f'

# (csv basename, panel title, [(cx, cy, r), ...] obstacles)
PANELS = [
    ('uniform_n200_s1.csv',     'Free space\n$N=200$',              []),
    ('gaussian_n100_s1.csv',    'Target region\n$N=100$',           []),
    ('cov_free_n60_r60_s1.csv', 'Free space\n$N=60$, $r=60$ m',     []),
    ('cov_free_n40_r90_s1.csv', 'Free space\n$N=40$, $r=90$ m',     []),
    ('cov_obs_n60_r60_s1.csv',  'Obstacle\n$N=60$, $r=60$ m', [(0.0, 0.0, 120.0)]),
    ('cov_obs_n40_r90_s1.csv',  'Obstacle\n$N=40$, $r=90$ m', [(0.0, 0.0, 120.0)]),
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
    ax.set_xticks([-200, 0, 200])
    ax.set_yticks([-200, 0, 200])
    ax.set_title(f'{label} {title}')


def generate(gen_dir=GEN, out_dirs=(PAPER_FIG, REPO_FIG)):
    # Printed at full \linewidth (~6.3 in) as a 2x3 grid; render only slightly
    # larger so the ~0.75x downscale keeps labels legible.
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
    for ax, (csv, title, circles), label in zip(axes.ravel(), PANELS, LABELS):
        _panel(ax, gen_dir, csv, title, circles, label)
    fig.tight_layout()
    written = []
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, 'deployment_scenarios.png')
        fig.savefig(out, dpi=DPI)
        written.append(out)
        print(f'wrote {out}')
    plt.close(fig)
    return written


if __name__ == '__main__':
    generate()
