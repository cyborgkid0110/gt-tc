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


def test_clustering_cost_uses_bs_distance():
    """calc_node_cost(clustering=True) measures tx distance to the BS, so a node
    sitting on the BS costs less than the same node with a far-away BS."""
    on_bs = _net((30.0, 40.0)).calc_node_cost(
        _net((30.0, 40.0)).sensors[0], 'CM', clustering=True)
    far_bs = _net((0.0, 0.0)).calc_node_cost(
        _net((0.0, 0.0)).sensors[0], 'CM', clustering=True)
    assert far_bs > on_bs, "tx-to-BS cost should grow with BS distance"
    print("  clustering cost uses BS distance OK")


def _grid_net(bs_pos):
    """A small connected cluster of nodes near the origin."""
    sensors = [Sensor(id=i, x=float(x), y=float(y), e0=0.005, power=2.5e-4,
                      Vpre=3.0)
               for i, (x, y) in enumerate([(0, 0), (20, 0), (0, 20), (20, 20)])]
    return NetworkModel(sensors, 250, p_max=2.5e-4, bs_pos=bs_pos)


def test_connectivity_requires_reachable_bs():
    """A node-connected layout is still infeasible if no node can reach the BS."""
    near = _grid_net((10.0, 10.0))
    assert near.check_potential_connectivity(), "BS amid the cluster is reachable"
    far = _grid_net((10000.0, 10000.0))
    assert not far.check_potential_connectivity(), "unreachable BS must fail"
    print("  connectivity requires reachable BS OK")


if __name__ == '__main__':
    test_default_bs_is_origin()
    test_offset_bs()
    test_clustering_cost_uses_bs_distance()
    test_connectivity_requires_reachable_bs()
    print("\nAll bs-param tests passed.")
