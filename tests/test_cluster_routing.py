"""Unit tests for cluster-aware routing tree + energy.

Run:  conda run -n base python tests/test_cluster_routing.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Sensor, NetworkModel


def make_net(positions, vpre=3.0, area=250.0, **params):
    """Tiny NetworkModel with sensors at given (x, y); rc set per-test."""
    sensors = [Sensor(id=i, x=x, y=y, e0=1.0, power=0.0001, Vpre=vpre)
               for i, (x, y) in enumerate(positions)]
    return NetworkModel(sensors, area, **params)


def _set_rc(net, rc):
    for s in net.sensors:
        s.rc = rc


def test_gateway_ch_with_members():
    # CH near BS (gateway) + two members.
    net = make_net([(10, 0), (15, 0), (10, 5)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    ch, m1, m2 = net.sensors
    ch.is_ch = True
    m1.ch_belong = ch
    m2.ch_belong = ch
    tree = net.build_cluster_routing_tree()

    assert tree[0]['parent_id'] is None          # gateway -> BS
    assert tree[0]['depth'] == 0
    assert tree[0]['delivers'] is True
    assert math.isclose(tree[0]['tx_dist'], 10.0)
    assert tree[1]['parent_id'] == 0 and tree[1]['depth'] == 1
    assert tree[1]['delivers'] is True
    assert math.isclose(tree[1]['tx_dist'], 5.0)  # dist (15,0)-(10,0)
    assert tree[2]['parent_id'] == 0 and tree[2]['delivers'] is True


def test_three_ch_backbone_to_gateway():
    # A(gateway)-B-C chain; only A reaches BS.
    net = make_net([(30, 0), (70, 0), (110, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    a, b, c = net.sensors
    for s in (a, b, c):
        s.is_ch = True
    a.ch_neighbors = [b]
    b.ch_neighbors = [a, c]
    c.ch_neighbors = [b]
    tree = net.build_cluster_routing_tree()

    assert tree[0]['depth'] == 0 and tree[0]['parent_id'] is None   # gateway
    assert tree[1]['depth'] == 1 and tree[1]['parent_id'] == 0
    assert tree[2]['depth'] == 2 and tree[2]['parent_id'] == 1
    assert all(tree[i]['delivers'] for i in (0, 1, 2))
    assert math.isclose(tree[1]['tx_dist'], 40.0)
    assert math.isclose(tree[2]['tx_dist'], 40.0)


def test_stranded_component_no_gateway():
    # Two CHs, both beyond rc of BS -> stranded; terminal = closest to BS.
    net = make_net([(120, 0), (160, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    a, b = net.sensors
    a.is_ch = b.is_ch = True
    a.ch_neighbors = [b]
    b.ch_neighbors = [a]
    tree = net.build_cluster_routing_tree()

    cap = net.calc_comm_range(net.p_max)
    assert tree[0]['parent_id'] is None and tree[0]['depth'] == 0   # terminal (closer)
    assert tree[0]['delivers'] is False
    assert math.isclose(tree[0]['tx_dist'], min(120.0, cap))
    assert tree[1]['parent_id'] == 0 and tree[1]['delivers'] is False
    assert math.isclose(tree[1]['tx_dist'], 40.0)


def test_orphan_cm():
    # No CHs: a near orphan delivers, a far orphan does not.
    net = make_net([(10, 0), (200, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    near, far = net.sensors
    tree = net.build_cluster_routing_tree()

    cap = net.calc_comm_range(net.p_max)
    assert tree[0]['delivers'] is True and tree[0]['parent_id'] is None
    assert math.isclose(tree[0]['tx_dist'], 10.0)
    assert tree[1]['delivers'] is False
    assert math.isclose(tree[1]['tx_dist'], min(200.0, cap))


def test_multihop_cm():
    # Gateway CH - relay member - leaf member; leaf is out of CH range and
    # reaches the CH only by relaying through the middle member.
    net = make_net([(10, 0), (50, 0), (90, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    ch, relay, leaf = net.sensors
    ch.is_ch = True
    relay.ch_belong = ch
    leaf.ch_belong = ch
    tree = net.build_cluster_routing_tree()

    assert tree[0]['parent_id'] is None and tree[0]['depth'] == 0   # gateway
    assert tree[1]['parent_id'] == 0 and tree[1]['depth'] == 1      # direct child
    assert math.isclose(tree[1]['tx_dist'], 40.0)
    assert tree[2]['parent_id'] == 1 and tree[2]['depth'] == 2      # relayed
    assert math.isclose(tree[2]['tx_dist'], 40.0)
    assert all(tree[i]['delivers'] for i in (0, 1, 2))
    # forward-only relay accounting: middle member carries one descendant.
    assert tree[1]['num_descendants'] == 1
    assert tree[2]['num_descendants'] == 0


def test_no_path_cm():
    # Member assigned to a CH it cannot reach (no direct or relayed path).
    net = make_net([(10, 0), (200, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    ch, m = net.sensors
    ch.is_ch = True
    m.ch_belong = ch
    tree = net.build_cluster_routing_tree()

    assert tree[0]['parent_id'] is None and tree[0]['delivers'] is True  # gateway
    assert tree[1]['parent_id'] is None                                  # no path
    assert tree[1]['delivers'] is False
    assert math.isclose(tree[1]['tx_dist'], m.rc)   # charged at its own power


def test_dead_ch_member_becomes_orphan():
    # Member assigned to a CH that is NOT alive -> falls to the orphan branch.
    # is_alive is a property (e_res > 0), so kill the CH via e_res = 0.
    net = make_net([(10, 0), (15, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    ch, m = net.sensors
    ch.is_ch = True
    m.ch_belong = ch
    ch.e_res = 0.0                # dead CH (is_alive -> False)
    assert ch.is_alive is False
    tree = net.build_cluster_routing_tree()

    assert 0 not in tree          # dead CH excluded from the tree
    # Member's ch_belong is dead -> orphan gateway; delivers by own dist to BS.
    assert tree[1]['parent_id'] is None and tree[1]['depth'] == 0
    expected_delivers = net.dist_to_bs(m) <= m.rc          # 15 <= 50 -> True
    assert tree[1]['delivers'] is expected_delivers
    assert tree[1]['delivers'] is True
    assert math.isclose(tree[1]['tx_dist'], 15.0)          # dist (15,0)-BS(0,0)


def test_two_clusters_share_backbone():
    # A(gateway) + B(backbone via A). Each CH has its own member; members route
    # only to their OWN CH (no cross-cluster relay). rc=50, BS at origin.
    #   A  (30,0): dist BS = 30 <= 50 -> gateway, depth 0
    #   B  (75,0): dist A  = 45 <= 50, dist BS = 75 > 50 -> backbone, depth 1
    #   mA (40,0): dist A  = 10 <= 50 -> child of A, depth 1
    #   mB (90,0): dist B  = 15 <= 50, dist A = 60 > 50 -> child of B, depth 2
    net = make_net([(30, 0), (75, 0), (40, 0), (90, 0)])
    _set_rc(net, 50.0)
    net.discover_neighbors()
    a, b, ma, mb = net.sensors
    a.is_ch = b.is_ch = True
    a.ch_neighbors = [b]
    b.ch_neighbors = [a]
    ma.ch_belong = a
    mb.ch_belong = b
    tree = net.build_cluster_routing_tree()

    # CH backbone
    assert tree[0]['parent_id'] is None and tree[0]['depth'] == 0   # A gateway
    assert tree[1]['parent_id'] == 0 and tree[1]['depth'] == 1      # B via A
    # Members attach only to their own CH
    assert tree[2]['parent_id'] == 0 and tree[2]['depth'] == 1      # mA -> A
    assert tree[3]['parent_id'] == 1 and tree[3]['depth'] == 2      # mB -> B
    # All deliver
    assert all(tree[i]['delivers'] for i in (0, 1, 2, 3))
    # tx distances
    assert math.isclose(tree[0]['tx_dist'], 30.0)   # A -> BS
    assert math.isclose(tree[1]['tx_dist'], 45.0)   # B -> A
    assert math.isclose(tree[2]['tx_dist'], 10.0)   # mA -> A
    assert math.isclose(tree[3]['tx_dist'], 15.0)   # mB -> B


def test_cluster_costs_deterministic_with_zero_vpre():
    # Vpre=0 -> sensing/processing terms vanish -> costs are deterministic.
    net = make_net([(10, 0), (15, 0), (10, 5)], vpre=0.0)
    _set_rc(net, 50.0)
    net.discover_neighbors()
    ch, m1, m2 = net.sensors
    ch.is_ch = True
    m1.ch_belong = ch
    m2.ch_belong = ch
    tree = net.build_cluster_routing_tree()
    costs = net.compute_cluster_maintenance_costs(tree)

    # Leaf CM cost == one CM-role TX to its CH (nd=0).
    exp_m1 = net.calc_tx_cost(tree[1]['tx_dist'], 'CM')
    assert abs(costs[1] - exp_m1) < 1e-18
    # CH cost == RX(2 member raw packets, m_pkt_s each) + aggregation
    #            + one CH-role TX to BS.  (nd_ch = 0 here.)
    exp_ch = (2 * net.m_pkt_s * net.e_elec
              + net.m_pkt_l * net.e_agg
              + net.calc_tx_cost(tree[0]['tx_dist'], 'CH'))
    assert abs(costs[0] - exp_ch) < 1e-18


def test_cluster_costs_forward_only_relay_with_zero_vpre():
    # Gateway CH - relay member - leaf member; relay forwards the leaf packet.
    net = make_net([(10, 0), (50, 0), (90, 0)], vpre=0.0)
    _set_rc(net, 50.0)
    net.discover_neighbors()
    ch, relay, leaf = net.sensors
    ch.is_ch = True
    relay.ch_belong = ch
    leaf.ch_belong = ch
    tree = net.build_cluster_routing_tree()
    costs = net.compute_cluster_maintenance_costs(tree)

    # Leaf: one CM TX (nd=0).
    assert abs(costs[2] - net.calc_tx_cost(tree[2]['tx_dist'], 'CM')) < 1e-18
    # Relay: forward own + 1 descendant => 2 CM TX + 1 RX of m_pkt_s (nd=1).
    exp_relay = (2 * net.calc_tx_cost(tree[1]['tx_dist'], 'CM')
                 + 1 * net.m_pkt_s * net.e_elec)
    assert abs(costs[1] - exp_relay) < 1e-18
    # CH: receives both member raw packets (M_ch=2, m_pkt_s each), nd_ch=0.
    exp_ch = (2 * net.m_pkt_s * net.e_elec
              + net.m_pkt_l * net.e_agg
              + net.calc_tx_cost(tree[0]['tx_dist'], 'CH'))
    assert abs(costs[0] - exp_ch) < 1e-18


def test_cluster_costs_backbone_forward_only_with_zero_vpre():
    # A(gateway)-B-C CH chain, no members. Forward-only backbone: C sends 1 pkt;
    # B forwards C's + its own (2 TX, 1 RX); A forwards both + its own (3 TX, 2 RX).
    net = make_net([(30, 0), (70, 0), (110, 0)], vpre=0.0)
    _set_rc(net, 50.0)
    net.discover_neighbors()
    a, b, c = net.sensors
    for s in (a, b, c):
        s.is_ch = True
    a.ch_neighbors = [b]
    b.ch_neighbors = [a, c]
    c.ch_neighbors = [b]
    tree = net.build_cluster_routing_tree()
    costs = net.compute_cluster_maintenance_costs(tree)

    e = net.m_pkt_l * net.e_elec
    agg = net.m_pkt_l * net.e_agg
    txC = net.calc_tx_cost(tree[2]['tx_dist'], 'CH')
    txB = net.calc_tx_cost(tree[1]['tx_dist'], 'CH')
    txA = net.calc_tx_cost(tree[0]['tx_dist'], 'CH')
    assert abs(costs[2] - (agg + txC)) < 1e-18              # nd_ch=0
    assert abs(costs[1] - (1 * e + agg + 2 * txB)) < 1e-18  # nd_ch=1
    assert abs(costs[0] - (2 * e + agg + 3 * txA)) < 1e-18  # nd_ch=2


def test_metric_avg_hop_none_when_no_delivery():
    from metrics import MetricsCollector
    net = make_net([(0, 0), (0, 0), (0, 0)])   # 3 alive sensors
    mc = MetricsCollector(net)

    # Round 0: node 0 delivers at depth 2 (=> 3 hops), node 1 stranded.
    mc.record_round(net, 0, {}, routing_tree={
        0: {'depth': 2, 'delivers': True},
        1: {'depth': 0, 'delivers': False},
    })
    # Round 1: nothing delivers -> avg_hop None, delivered 0.
    mc.record_round(net, 1, {}, routing_tree={
        0: {'depth': 0, 'delivers': False},
    })
    ts = mc.time_series()
    assert ts['delivered'] == [1, 0]
    assert ts['avg_hop'][0] == 3.0          # mean(depth)=2, +1
    assert ts['avg_hop'][1] is None
    summ = mc.finalize()
    assert abs(summ['mean_avg_hop'] - 3.0) < 1e-12   # None excluded


if __name__ == '__main__':
    test_gateway_ch_with_members()
    test_three_ch_backbone_to_gateway()
    test_stranded_component_no_gateway()
    test_orphan_cm()
    test_multihop_cm()
    test_no_path_cm()
    test_dead_ch_member_becomes_orphan()
    test_two_clusters_share_backbone()
    test_cluster_costs_deterministic_with_zero_vpre()
    test_cluster_costs_forward_only_relay_with_zero_vpre()
    test_cluster_costs_backbone_forward_only_with_zero_vpre()
    test_metric_avg_hop_none_when_no_delivery()
    print('Task 1 tests passed')
