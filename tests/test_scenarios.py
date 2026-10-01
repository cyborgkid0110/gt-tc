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


def test_single_node():
    """A 1-node scenario round-trips (genfromtxt returns a 0-d array)."""
    positions = np.array([[5.0, -7.5]])
    vpre = np.array([3.3])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'one.csv')
        save_scenario(path, positions, vpre, (0.0, 0.0),
                      {'source': 'uniform', 'num_nodes': 1, 'seed': 0,
                       'area': 250, 'coverage_radius': ''})
        pos2, vpre2, bs2, meta2 = load_scenario(path)
    assert pos2.shape == (1, 2) and vpre2.shape == (1,)
    assert np.allclose(positions, pos2) and np.allclose(vpre, vpre2)
    print("  single-node OK")


def test_build_network_from_scenario():
    """A frozen scenario reproduces a NetworkModel with the stored state."""
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


if __name__ == '__main__':
    test_round_trip()
    test_single_node()
    test_build_network_from_scenario()
    print("\nAll scenarios tests passed.")
