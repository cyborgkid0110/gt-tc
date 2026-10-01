# tests/test_regions.py
"""Region geometry: deployable area, obstacle exclusion, path constraint, BS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.regions import Region


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


def test_sample_and_project_in_region():
    import numpy as np
    r = Region.from_def(_square_obstacle_def())
    rng = np.random.default_rng(0)
    pts = r.sample_inside(25, rng)
    assert pts.shape == (25, 2)
    assert all(r.contains(p) for p in pts), "sampled points must be deployable"
    # projecting a point inside the obstacle lands on the deployable boundary
    proj = r.project((0.0, 0.0))
    assert r.deployable.distance(__import__('shapely').geometry.Point(proj)) < 1e-6
    print("  sample + project OK")


def test_coverage_targets_in_region():
    r = Region.from_def(_square_obstacle_def())
    targets = r.coverage_targets(40)
    assert len(targets) > 0
    assert all(r.contains(t) for t in targets), "targets must be deployable"
    print("  coverage targets OK")


def test_circle_obstacle_excluded():
    d = {'area': 250, 'num_nodes': 10, 'coverage_radius': 40, 'seed': 0,
         'bs': [180.0, -180.0], 'obstacles': [],
         'circles': [[0.0, 0.0, 80.0]], 'paths': []}
    r = Region.from_def(d)
    assert not r.contains((0.0, 0.0)), "centre is inside the circle obstacle"
    assert not r.contains((50.0, 0.0)), "inside the circle radius"
    assert r.contains((180.0, -180.0)), "BS location must be deployable"
    print("  circle obstacle exclusion OK")


if __name__ == '__main__':
    test_obstacle_excluded()
    test_bs_in_obstacle_rejected()
    test_path_constrains_placement()
    test_sample_and_project_in_region()
    test_coverage_targets_in_region()
    test_circle_obstacle_excluded()
    print("\nAll region tests passed.")
