"""Render node-deployment scenarios.

Two modes:

  # default — the five random baselines for documentation (BS at origin)
  conda run -n base python plot_deployments.py            -> docs/figures/*.png

  # a single custom scenario, with obstacle/path geometry and an off-centre BS
  conda run -n base python plot_deployments.py --def scenarios/defs/example.yaml
  conda run -n base python plot_deployments.py --scenario scenarios/gen/uniform_n200_s7.csv

In the default mode, positions come from ``build_network`` so the figure reflects
the *feasibility-resampled* layout the sweep actually runs. The single-scenario
mode draws the deployable area, obstacles, paths, coverage disks, and the base
station wherever the scenario places it.

Single-scenario options:
  --coverage-disks / --no-coverage-disks   shade each node's coverage_radius disk
                     (default: on for --def, off for --scenario)
  --show-links       draw edges between nodes within the physics max-power comm
                     range (a connectivity preview)
  --out PATH         output PNG (default docs/figures/scenario_<name>.png)
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
import numpy as np
import yaml

from main import build_network
from scenarios import load_scenario

NUM_NODES = 200
# Per-deployment node-count overrides; mirrors benchmark.py DEPLOYMENT_NODES so
# the rendered figure matches what the sweep actually runs.
DEPLOYMENT_NODES = {'gaussian': 100}
AREA = 250
SEED = 1
FIG_DIR = os.path.join('docs', 'figures')

NODE_COLOR = '#1f77b4'
DEPLOYABLE_COLOR = '#cfe8ff'
OBSTACLE_COLOR = '#d9534f'


def nodes_for(deployment):
    """Node count for a deployment, honouring DEPLOYMENT_NODES overrides."""
    return DEPLOYMENT_NODES.get(deployment, NUM_NODES)

# Scenario render order + a one-line caption shown as the subplot title.
SCENARIOS = [
    ('poisson', 'Poisson-disk (blue-noise, even spacing)'),
    ('uniform', 'Uniform i.i.d. (baseline)'),
    ('grid', 'Square lattice'),
    ('gaussian', 'Gaussian blobs (hotspots)'),
    ('edge', 'Edge-biased (energy-hole stressor)'),
]


# --------------------------------------------------------------------------- #
#  Default mode: the five random baselines                                    #
# --------------------------------------------------------------------------- #

def _scatter(ax, pts, title):
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c='#1f77b4', alpha=0.8,
               edgecolors='none')
    ax.scatter([0], [0], marker='*', s=180, c='red', zorder=3, label='Base station')
    ax.set_xlim(-AREA, AREA)
    ax.set_ylim(-AREA, AREA)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def generate(fig_dir=FIG_DIR):
    os.makedirs(fig_dir, exist_ok=True)

    # Combined figure (2x3 grid; last cell hosts the legend).
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (name, caption) in zip(axes, SCENARIOS):
        n = nodes_for(name)
        # Same feasibility-resampled layout the benchmark runs for (name, n, SEED).
        net = build_network(name, n, SEED)
        pts = np.array([[s.x, s.y] for s in net.sensors], dtype=float)
        _scatter(ax, pts, f'{name} (N={n}) — {caption}')

        # Per-scenario standalone PNG too (reuse the identical positions).
        f1, a1 = plt.subplots(figsize=(5, 5))
        _scatter(a1, pts, f'{name} (N={n})')
        a1.legend(fontsize=7, loc='upper right')
        f1.tight_layout()
        f1.savefig(os.path.join(fig_dir, f'deployment_{name}.png'), dpi=120)
        plt.close(f1)

    # Use the spare 6th cell for a shared legend, then hide its axes.
    axes[-1].scatter([], [], s=8, c='#1f77b4', label='Sensor node')
    axes[-1].scatter([], [], marker='*', s=180, c='red', label='Base station')
    axes[-1].legend(loc='center', fontsize=11, frameon=False)
    axes[-1].axis('off')

    overrides = ', '.join(f'{k} N={v}' for k, v in DEPLOYMENT_NODES.items())
    extra = f', {overrides}' if overrides else ''
    fig.suptitle(f'Node-deployment scenarios (N={NUM_NODES}{extra}, '
                 f'area=±{AREA}, seed={SEED})', fontsize=12)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'deployments.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'Deployment figures written to {fig_dir}')


# --------------------------------------------------------------------------- #
#  Single-scenario mode: geometry + coverage + off-centre BS                  #
# --------------------------------------------------------------------------- #

def _polygon_patch(geom, **kw):
    """A matplotlib PathPatch for a shapely (Multi)Polygon, holes included.

    shapely rings are already closed (last coord repeats the first), so each ring
    of n coords needs exactly n path codes: one MOVETO then LINETOs.
    """
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    verts, codes = [], []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            verts.extend(coords)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(coords) - 1))
    return PathPatch(Path(verts, codes), **kw)


def _r_conn():
    """Physics max-power comm range (the benchmark's connectivity radius)."""
    from main import (P_MAX, SNR, NF_RX, N0, BW, WAVE, GAMMA, G_TX, G_RX, ETA,
                      R_BIT, E0)
    from model import Sensor, NetworkModel
    s = Sensor(id=0, x=0.0, y=0.0, e0=E0, power=P_MAX, Vpre=3.0)
    net = NetworkModel([s], AREA, snr=SNR, nf_rx=NF_RX, n0=N0, bw=BW, wave=WAVE,
                       gamma=GAMMA, g_tx=G_TX, g_rx=G_RX, eta=ETA, r_bit=R_BIT, p_max=P_MAX)
    return net.calc_comm_range(P_MAX)


def _draw_links(ax, positions, bs_pos, r_conn):
    pts = np.vstack([np.asarray(bs_pos, dtype=float), positions])
    idx = np.arange(len(pts))
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        for j in np.nonzero((d <= r_conn) & (idx > i))[0]:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    c='#777777', lw=0.4, alpha=0.5, zorder=3)


def plot_from_def(def_path, coverage_disks, show_links, out):
    from scenarios.regions import Region
    with open(def_path) as f:
        d = yaml.safe_load(f)
    region = Region.from_def(d)
    area = float(d['area'])
    name = d['name']
    cov_r = float(d['coverage_radius'])

    csv = os.path.join('scenarios', 'gen', f'{name}.csv')
    if not os.path.exists(csv):
        raise SystemExit(
            f"No frozen CSV at {csv}. Generate it first:\n"
            f"  conda run -n base python -m scenarios.make_coverage_scenario --def {def_path}")
    positions, _vpre, bs_pos, _meta = load_scenario(csv)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(_polygon_patch(region.deployable, facecolor=DEPLOYABLE_COLOR,
                                edgecolor='#3a7bd5', lw=1.0, zorder=1,
                                label='Deployable area'))
    for ring in d.get('obstacles', []):
        xs, ys = zip(*ring)
        ax.fill(xs, ys, facecolor=OBSTACLE_COLOR, alpha=0.55, hatch='xx',
                edgecolor=OBSTACLE_COLOR, lw=1.0, zorder=2)
    for cx, cy, rad in d.get('circles', []):
        ax.add_patch(Circle((cx, cy), rad, facecolor=OBSTACLE_COLOR, alpha=0.55,
                            hatch='xx', edgecolor=OBSTACLE_COLOR, lw=1.0, zorder=2))

    if coverage_disks:
        for x, y in positions:
            ax.add_patch(Circle((x, y), cov_r, facecolor='#2ca02c', alpha=0.10,
                                edgecolor='none', zorder=2))
    if show_links:
        _draw_links(ax, positions, bs_pos, _r_conn())

    _finish(ax, positions, bs_pos, area,
            f'{name}: {len(positions)} nodes, coverage r={cov_r:g} m')
    return _save(fig, out, name)


def plot_from_scenario(csv_path, coverage_disks, show_links, out):
    positions, _vpre, bs_pos, meta = load_scenario(csv_path)
    area = float(meta.get('area', AREA))
    name = meta.get('source', os.path.splitext(os.path.basename(csv_path))[0])
    cov_r = meta.get('coverage_radius', '')

    fig, ax = plt.subplots(figsize=(7, 7))
    if coverage_disks and cov_r not in ('', None):
        for x, y in positions:
            ax.add_patch(Circle((x, y), float(cov_r), facecolor='#2ca02c',
                                alpha=0.10, edgecolor='none', zorder=2))
    if show_links:
        _draw_links(ax, positions, bs_pos, _r_conn())

    _finish(ax, positions, bs_pos, area, f'{name}: {len(positions)} nodes')
    return _save(fig, out, name)


def _obstacles_for_tag(tag):
    """Circular obstacles [[cx, cy, r], ...] for a coverage tag, if known.

    Frozen CSVs store only node positions + BS, not obstacle geometry, so look
    the circles up from the coverage-config table by tag. Returns [] for the
    random deployments (no obstacles) or any unrecognised tag.
    """
    try:
        from scenarios.freeze_coverage_scenarios import COVERAGE_CONFIGS
    except Exception:
        return []
    for c in COVERAGE_CONFIGS:
        if c.get('tag') == tag:
            return c.get('circles', [])
    return []


def plot_grid(csv_paths, coverage_disks, rows, cols, out, titles=None):
    """Tile several frozen scenario CSVs into one rows x cols figure.

    One panel per CSV (nodes + BS + any known obstacle circles). The panel title
    defaults to the scenario tag (CSV meta 'source', else the filename stem minus
    _s<seed>); pass `titles` (one per CSV, in order) to override the captions.
    """
    n = len(csv_paths)
    if n > rows * cols:
        raise SystemExit(f"{n} scenarios do not fit in a {rows}x{cols} grid")
    if titles is not None and len(titles) != n:
        raise SystemExit(f"--titles has {len(titles)} entries but {n} scenarios")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 4.3))
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, csv_path) in enumerate(zip(axes, csv_paths)):
        positions, _vpre, bs_pos, meta = load_scenario(csv_path)
        area = float(meta.get('area', AREA))
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        # Normalise to the family tag: drop a trailing _s<seed> (some CSVs record
        # 'source' with the seed, some without) so the obstacle lookup matches.
        tag = re.sub(r'_s\d+$', '', meta.get('source') or stem)
        cov_r = meta.get('coverage_radius', '')

        for cx, cy, rad in _obstacles_for_tag(tag):
            ax.add_patch(Circle((cx, cy), rad, facecolor=OBSTACLE_COLOR,
                                alpha=0.5, hatch='xx', edgecolor=OBSTACLE_COLOR,
                                lw=1.0, zorder=2))
        if coverage_disks and cov_r not in ('', None):
            for x, y in positions:
                ax.add_patch(Circle((x, y), float(cov_r), facecolor='#2ca02c',
                                    alpha=0.08, edgecolor='none', zorder=2))
        ax.scatter(positions[:, 0], positions[:, 1], s=10, c=NODE_COLOR,
                   edgecolors='white', linewidths=0.2, zorder=4)
        ax.scatter([bs_pos[0]], [bs_pos[1]], marker='*', s=160, c='red',
                   edgecolors='black', linewidths=0.4, zorder=5)
        ax.set_xlim(-area, area)
        ax.set_ylim(-area, area)
        ax.set_aspect('equal')
        title = titles[i] if titles is not None else f'{tag} (N={len(positions)})'
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:           # hide any spare cells
        ax.axis('off')

    handles = [
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=NODE_COLOR,
                   markersize=6, label='Sensor node'),
        plt.Line2D([], [], marker='*', color='w', markerfacecolor='red',
                   markersize=12, label='Base station'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
               frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    if out is None:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, 'scenarios_grid.png')
    else:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'wrote {out}')
    return out


def _finish(ax, positions, bs_pos, area, title):
    ax.scatter(positions[:, 0], positions[:, 1], s=14, c=NODE_COLOR,
               edgecolors='white', linewidths=0.3, zorder=4, label='Sensor node')
    ax.scatter([bs_pos[0]], [bs_pos[1]], marker='*', s=260, c='red',
               edgecolors='black', linewidths=0.5, zorder=5, label='Base station')
    ax.set_xlim(-area, area)
    ax.set_ylim(-area, area)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.tick_params(labelsize=8)


def _save(fig, out, name):
    if out is None:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, f'scenario_{name}.png')
    else:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'wrote {out}')
    return out


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--def', dest='defn', help='region-definition YAML '
                       '(single-scenario mode)')
    group.add_argument('--scenario', help='frozen scenario CSV (single-scenario mode)')
    group.add_argument('--grid', nargs='+', metavar='CSV',
                       help='tile several frozen scenario CSVs into one figure')
    ap.add_argument('--rows', type=int, default=2, help='grid rows (default 2)')
    ap.add_argument('--cols', type=int, default=3, help='grid cols (default 3)')
    ap.add_argument('--titles', nargs='+', metavar='TITLE',
                    help='custom panel titles for --grid (one per CSV, in order)')
    ap.add_argument('--coverage-disks', dest='disks', action='store_true',
                    default=None, help='shade coverage_radius disks')
    ap.add_argument('--no-coverage-disks', dest='disks', action='store_false',
                    help='do not shade coverage disks')
    ap.add_argument('--show-links', action='store_true',
                    help='draw max-power connectivity edges')
    ap.add_argument('--out', default=None, help='output PNG path (single-scenario mode)')
    args = ap.parse_args()

    if args.defn:
        disks = True if args.disks is None else args.disks
        plot_from_def(args.defn, disks, args.show_links, args.out)
    elif args.scenario:
        disks = False if args.disks is None else args.disks
        plot_from_scenario(args.scenario, disks, args.show_links, args.out)
    elif args.grid:
        disks = False if args.disks is None else args.disks
        plot_grid(args.grid, disks, args.rows, args.cols, args.out, args.titles)
    else:
        generate()


if __name__ == '__main__':
    main()
