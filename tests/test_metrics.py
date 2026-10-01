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
        # a node "delivers" iff it is alive and currently reachable;
        # depth = node id (deterministic, for avg_hop assertions)
        return {sid: {'depth': sid} for sid in self._reachable
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
    assert 'avg_tx_power' not in m, m   # now a universal-core metric, not a family extra
    assert m['lambda2'] is None, m
    print('  topology family OK')


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
    assert ts['avg_hop'][1] is None, ts['avg_hop']   # empty tree -> None

    s = mc.finalize()
    # None rounds excluded from the mean: mean([2.5]) = 2.5
    assert abs(s['mean_avg_hop'] - 2.5) < 1e-9, s
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


if __name__ == '__main__':
    test_lifetime_markers()
    test_pdr_and_energy_per_packet()
    test_energy_per_packet_none_when_no_delivery()
    test_clustering_family_metrics()
    test_topology_family_metrics()
    test_avg_hop_series_and_summary()
    test_avg_tx_power_universal()
    print('\nAll metrics tests passed.')
