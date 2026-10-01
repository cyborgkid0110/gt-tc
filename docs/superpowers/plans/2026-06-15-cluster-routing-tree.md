# Cluster-Aware Routing Tree & Energy Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the 7 clustering protocols' per-round energy *and* hop/PDR metric along the real cluster data path (CM→multi-hop intra-cluster relay→CH→multi-hop CH backbone→BS) via a shared `build_cluster_routing_tree`, instead of the shortest physical path. Members reach their CH over a per-cluster reverse-BFS tree of same-cluster member links (forward-only relays, CH is the sole aggregator); a member with no path to its CH still pays a failed TX at its own power.

**Architecture:** Two new `NetworkModel` methods (`build_cluster_routing_tree`, `compute_cluster_maintenance_costs`) produce a cluster-aware tree and its energy. Each clustering algo builds the tree once in its maintenance phase, stashes it on `self._routing_tree`, and `BaseAlgorithm.run()` passes that exact tree to `MetricsCollector.record_round` so energy and metric are computed from the same tree (build-once-pass). Topology-control games keep `build_routing_tree` unchanged. `avg_hop` is recorded as `None` (not `0`) when nothing delivers.

**Tech Stack:** Python, NumPy, plain-script unit tests (`conda run -n base python tests/<file>.py`).

Spec: `docs/superpowers/specs/2026-06-15-cluster-routing-tree-design.md`

---

## File Structure

| File | Responsibility / change |
|------|-------------------------|
| `model.py` | Add `build_cluster_routing_tree`, `compute_cluster_maintenance_costs`; add `delivers` field to `build_routing_tree`. |
| `metrics.py` | `record_round(..., routing_tree=None)`; derive `delivered`/`avg_hop` from `delivers`; `avg_hop=None` on no-delivery; `finalize` skips `None`. |
| `algos/__init__.py` | `BaseAlgorithm`: `self._routing_tree`; `run()` resets + passes it; add `_charge_cluster_maintenance()` helper. |
| `algos/{gt2,leach,gtfr,fl_leach_pso,sca_levy}.py` | Replace maintenance body with `self._charge_cluster_maintenance()`. |
| `algos/{fc_cra,ee_tcm}.py` | Cluster maintenance with their extras (cluster-stable flag / CM compression). |
| `algos/{dia_mia,tcle,eftcg}.py` | Stash `self._routing_tree` (one line each); otherwise unchanged. |
| `tests/test_cluster_routing.py` | New unit tests for the builder + cost + metric. |
| `paper_tables.py` | Verify `None` handling (no change expected). |

---

## Task 1: `build_cluster_routing_tree` on NetworkModel

**Files:**
- Modify: `model.py` (add method after `build_routing_tree`, ~line 260)
- Test: `tests/test_cluster_routing.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cluster_routing.py`:

```python
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
    ch, m = net.sensors
    ch.is_ch = True
    m.ch_belong = ch
    tree = net.build_cluster_routing_tree()

    assert tree[0]['parent_id'] is None and tree[0]['delivers'] is True  # gateway
    assert tree[1]['parent_id'] is None                                  # no path
    assert tree[1]['delivers'] is False
    assert math.isclose(tree[1]['tx_dist'], m.rc)   # charged at its own power


if __name__ == '__main__':
    test_gateway_ch_with_members()
    test_three_ch_backbone_to_gateway()
    test_stranded_component_no_gateway()
    test_orphan_cm()
    test_multihop_cm()
    test_no_path_cm()
    print('Task 1 tests passed')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n base python tests/test_cluster_routing.py`
Expected: FAIL with `AttributeError: 'NetworkModel' object has no attribute 'build_cluster_routing_tree'`

- [ ] **Step 3: Implement `build_cluster_routing_tree`**

In `model.py`, add this method immediately after `build_routing_tree` (after its `return tree`, ~line 260):

```python
    def build_cluster_routing_tree(self):
        """Cluster routing tree: CM -> multi-hop intra-cluster relay -> CH ->
        multi-hop CH backbone -> BS.

        Same schema as build_routing_tree plus 'delivers' (bool): True iff the
        node's path actually reaches the BS. Members reach their CH over a
        per-cluster reverse-BFS of same-cluster member links; a member with no
        path to its CH gets parent_id=None, tx_dist=rc, delivers=False (it still
        pays a failed TX at its own power). Stranded clusters (no gateway CH)
        still get parents/tx_dist but delivers=False; the component's
        closest-to-BS CH makes a capped best-effort BS attempt.
        """
        from collections import deque

        chs = [s for s in self.sensors if s.is_alive and s.is_ch]
        ch_ids = {ch.id for ch in chs}

        # CH-only adjacency from ch_neighbors (alive CHs only), symmetrised.
        ch_adj = {cid: set() for cid in ch_ids}
        for ch in chs:
            for nb in ch.ch_neighbors:
                if nb.is_alive and nb.is_ch and nb.id in ch_ids:
                    ch_adj[ch.id].add(nb.id)
                    ch_adj[nb.id].add(ch.id)

        parent, depth, tx_dist, delivers = {}, {}, {}, {}
        cap = self.calc_comm_range(self.p_max)

        # Gateway CHs + reachable backbone (reverse-BFS from gateways).
        queue = deque()
        for ch in chs:
            if self.dist_to_bs(ch) <= ch.rc:
                parent[ch.id] = None
                depth[ch.id] = 0
                tx_dist[ch.id] = self.dist_to_bs(ch)
                delivers[ch.id] = True
                queue.append(ch.id)
        while queue:
            cur = queue.popleft()
            for nb in ch_adj[cur]:
                if nb not in parent:
                    parent[nb] = cur
                    depth[nb] = depth[cur] + 1
                    tx_dist[nb] = self.sensors[nb].distance_to(self.sensors[cur])
                    delivers[nb] = True
                    queue.append(nb)

        # Stranded CH components (no gateway). Root at closest-to-BS terminal.
        stranded = [cid for cid in ch_ids if cid not in parent]
        stranded_set = set(stranded)
        visited = set()
        for start in stranded:
            if start in visited:
                continue
            comp, q = [], deque([start])
            visited.add(start)
            while q:
                c = q.popleft()
                comp.append(c)
                for nb in ch_adj[c]:
                    if nb in stranded_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comp_set = set(comp)
            terminal = min(comp, key=lambda i: self.dist_to_bs(self.sensors[i]))
            parent[terminal] = None
            depth[terminal] = 0
            tx_dist[terminal] = min(self.dist_to_bs(self.sensors[terminal]), cap)
            delivers[terminal] = False
            seen, q = {terminal}, deque([terminal])
            while q:
                cur = q.popleft()
                for nb in ch_adj[cur]:
                    if nb in comp_set and nb not in seen:
                        seen.add(nb)
                        parent[nb] = cur
                        depth[nb] = depth[cur] + 1
                        tx_dist[nb] = self.sensors[nb].distance_to(self.sensors[cur])
                        delivers[nb] = False
                        q.append(nb)

        # Member layer: multi-hop CM -> ... -> CH over same-cluster member edges
        # (forward-only). Reverse-BFS rooted at each CH already in the tree.
        members_by_ch = {}
        for s in self.sensors:
            if not s.is_alive or s.is_ch:
                continue
            ch = s.ch_belong
            if ch is not None and ch.is_alive and ch.is_ch and ch.id in parent:
                members_by_ch.setdefault(ch.id, []).append(s)

        for ch_id, members in members_by_ch.items():
            mids = {m.id for m in members}
            # rev[t] = same-cluster members m with directed edge m -> t.
            rev = {ch_id: []}
            for m in members:
                rev[m.id] = []
            for m in members:
                if self.edges[m.id, ch_id] == 1:
                    rev[ch_id].append(m.id)
                for t_id in mids:
                    if t_id != m.id and self.edges[m.id, t_id] == 1:
                        rev[t_id].append(m.id)
            q = deque([ch_id])
            seen = {ch_id}
            while q:
                cur = q.popleft()
                for m_id in rev[cur]:
                    if m_id not in seen:
                        seen.add(m_id)
                        parent[m_id] = cur
                        depth[m_id] = depth[cur] + 1
                        tx_dist[m_id] = self.sensors[m_id].distance_to(
                            self.sensors[cur])
                        delivers[m_id] = delivers[ch_id]
                        q.append(m_id)

        # Unreached members: no-path (live CH, no relay route) or orphan (no CH).
        for s in self.sensors:
            if not s.is_alive or s.is_ch or s.id in parent:
                continue
            ch = s.ch_belong
            if ch is not None and ch.is_alive and ch.is_ch and ch.id in parent:
                # Has a CH but no path to it -> failed TX at its own power.
                parent[s.id] = None
                depth[s.id] = 0
                tx_dist[s.id] = s.rc
                delivers[s.id] = False
            else:
                # Orphan: single-node CH for the round.
                reach = self.dist_to_bs(s) <= s.rc
                parent[s.id] = None
                depth[s.id] = 0
                tx_dist[s.id] = self.dist_to_bs(s) if reach \
                    else min(self.dist_to_bs(s), cap)
                delivers[s.id] = reach

        # num_descendants (subtree sizes), deepest first.
        descendants = {nid: 0 for nid in parent}
        for nid in sorted(parent, key=lambda x: depth[x], reverse=True):
            p = parent[nid]
            if p is not None:
                descendants[p] += 1 + descendants[nid]

        return {nid: {'parent_id': parent[nid], 'tx_dist': tx_dist[nid],
                      'depth': depth[nid], 'num_descendants': descendants[nid],
                      'delivers': delivers[nid]}
                for nid in parent}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n base python tests/test_cluster_routing.py`
Expected: `Task 1 tests passed`

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_cluster_routing.py
git commit -m "feat: add cluster-aware routing tree (CM->CH->backbone->BS)"
```

---

## Task 2: `compute_cluster_maintenance_costs` on NetworkModel

**Files:**
- Modify: `model.py` (add after `compute_maintenance_costs`, ~line 290)
- Test: `tests/test_cluster_routing.py` (add a test)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_cluster_routing.py` (before the `__main__` block):

```python
def test_cluster_costs_deterministic_with_zero_vpre():
    # Vpre=0 -> sensing/processing terms vanish -> costs are deterministic.
    net = make_net([(10, 0), (15, 0), (10, 5)], vpre=0.0)
    _set_rc(net, 50.0)
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
```

And add their calls in `__main__`:

```python
    test_cluster_costs_deterministic_with_zero_vpre()
    test_cluster_costs_forward_only_relay_with_zero_vpre()
    test_cluster_costs_backbone_forward_only_with_zero_vpre()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n base python tests/test_cluster_routing.py`
Expected: FAIL with `AttributeError: ... 'compute_cluster_maintenance_costs'`

- [ ] **Step 3: Implement `compute_cluster_maintenance_costs`**

In `model.py`, add immediately after `compute_maintenance_costs` (after its `return costs`, ~line 290):

```python
    def compute_cluster_maintenance_costs(self, tree):
        """Per-node energy along the cluster routing tree (forward-only members
        and forward-only CH backbone; each CH aggregates only its own cluster).

        Every node in the tree is charged regardless of 'delivers' (a stranded
        cluster still pays its transmissions). A member forwards its own packet
        plus every descendant's ((1+nd) CM TX + nd RX of m_pkt_s, no fusion); a
        leaf or no-path member has nd=0 (the no-path member pays one CM TX at
        tx_dist=rc). A CH receives every member raw packet of its cluster
        (m_pkt_s each), fuses them to one aggregated packet (e_agg), and on the
        backbone forwards that own packet plus every downstream CH's aggregated
        packet without re-fusing ((1+nd_ch) CH TX + nd_ch RX of m_pkt_l), where
        nd_ch = CH descendants below it. Nodes absent from the tree pay
        sensing+processing.
        """
        from collections import Counter

        m_ch = Counter()   # reached members per cluster CH (raw packets arriving)
        for nid, info in tree.items():
            s = self.sensors[nid]
            if not s.is_ch and info['parent_id'] is not None:  # reached member
                ch = s.ch_belong
                if ch is not None:
                    m_ch[ch.id] += 1

        # CH-only subtree sizes (downstream CHs forwarded through each CH),
        # deepest first over CH nodes; a CH's parent on the backbone is a CH.
        ch_desc = Counter()
        ch_nodes = [nid for nid in tree if self.sensors[nid].is_ch]
        for nid in sorted(ch_nodes, key=lambda x: tree[x]['depth'], reverse=True):
            p = tree[nid]['parent_id']
            if p is not None and self.sensors[p].is_ch:
                ch_desc[p] += 1 + ch_desc[nid]

        costs = {}
        for s in self.sensors:
            if not s.is_alive:
                continue
            m_bit = self.sensor_sample_bits
            i_sense = random.uniform(1e-8, 5e-7)
            c_sense = s.Vpre * i_sense * m_bit
            c_process = s.Vpre * m_bit * i_sense / 4

            if s.id not in tree:
                costs[s.id] = c_sense + c_process
                continue

            info = tree[s.id]
            if s.is_ch:
                nd_ch = ch_desc[s.id]
                rx = (m_ch[s.id] * self.m_pkt_s * self.e_elec
                      + nd_ch * self.m_pkt_l * self.e_elec)
                agg = self.m_pkt_l * self.e_agg
                tx = (1 + nd_ch) * self.calc_tx_cost(info['tx_dist'], 'CH')
                costs[s.id] = c_sense + c_process + rx + agg + tx
            else:
                nd = info['num_descendants']
                tx = (1 + nd) * self.calc_tx_cost(info['tx_dist'], 'CM')
                rx = nd * self.m_pkt_s * self.e_elec
                costs[s.id] = c_sense + c_process + tx + rx
        return costs
```

(`random` is already imported at the top of `model.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n base python tests/test_cluster_routing.py`
Expected: `Task 1 tests passed` (all five tests run, no assertion error)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_cluster_routing.py
git commit -m "feat: cluster maintenance energy (forward-only members, CH aggregation)"
```

---

## Task 3: Plumbing — build-once-pass, `delivers`, avg_hop=None

**Files:**
- Modify: `model.py` (`build_routing_tree`: add `delivers`)
- Modify: `metrics.py` (`record_round`, `finalize`)
- Modify: `algos/__init__.py` (`BaseAlgorithm`: `_routing_tree`, `run()`, helper)
- Test: `tests/test_cluster_routing.py` (metric test)

- [ ] **Step 1: Add `delivers=True` to `build_routing_tree`**

In `model.py`, in `build_routing_tree`, the per-node dict currently is:

```python
            tree[nid] = {
                'parent_id': pid,
                'tx_dist': tx_d,
                'depth': depth[nid],
                'num_descendants': descendants[nid],
            }
```

Change it to add `delivers` (every node in this tree reaches the BS):

```python
            tree[nid] = {
                'parent_id': pid,
                'tx_dist': tx_d,
                'depth': depth[nid],
                'num_descendants': descendants[nid],
                'delivers': True,
            }
```

- [ ] **Step 2: Write the failing metric test**

Append to `tests/test_cluster_routing.py` (before `__main__`):

```python
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
```

And add `test_metric_avg_hop_none_when_no_delivery()` to `__main__`.

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n base python tests/test_cluster_routing.py`
Expected: FAIL — `record_round()` got an unexpected keyword `routing_tree`.

- [ ] **Step 4: Update `metrics.py` `record_round` + `finalize`**

In `metrics.py`, change the `record_round` signature and the tree/avg_hop block.

Signature line:

```python
    def record_round(self, net, t, family_extras=None, routing_tree=None):
```

Replace the current tree/avg_hop block:

```python
        tree = net.build_routing_tree()
        delivered = len(tree)
        # avg_hop: mean depth over delivered nodes, +1 for the final hop to the
        # BS (a gateway has depth 0 = one hop to the sink). 0.0 if nothing
        # reaches the BS this round.
        avg_hop = (float(np.mean([n['depth'] for n in tree.values()])) + 1.0
                   if tree else 0.0)
```

with:

```python
        tree = routing_tree if routing_tree is not None \
            else net.build_routing_tree()
        delivering = [info for info in tree.values()
                      if info.get('delivers', True)]
        delivered = len(delivering)
        # avg_hop: mean depth over *delivering* nodes, +1 for the final hop to
        # the BS. None when nothing reaches the BS (undefined, not 0 hops).
        avg_hop = (float(np.mean([info['depth'] for info in delivering])) + 1.0
                   if delivering else None)
```

In `finalize`, replace:

```python
        mean_avg_hop = float(np.mean(self.avg_hop)) if self.avg_hop else 0.0
```

with (skip `None` no-delivery rounds):

```python
        _hops = [h for h in self.avg_hop if h is not None]
        mean_avg_hop = float(np.mean(_hops)) if _hops else 0.0
```

- [ ] **Step 5: Add plumbing to `BaseAlgorithm`**

In `algos/__init__.py`, in `BaseAlgorithm.__init__` (after `self.metrics = MetricsCollector(net)`), add:

```python
        self._routing_tree = None   # set by _run_round; passed to record_round
```

Replace the `run()` loop:

```python
    def run(self):
        """Main simulation loop."""
        while self.t < self.max_rounds:
            ok = self._run_round()
            if not ok:
                break
            # record before incrementing t so fnd matches t_no_dead's convention
            self.metrics.record_round(self.net, self.t,
                                      self._collect_family_metrics())
            self.t += 1

        self.metrics.finalize()
```

with:

```python
    def run(self):
        """Main simulation loop."""
        while self.t < self.max_rounds:
            self._routing_tree = None
            ok = self._run_round()
            if not ok:
                break
            # record before incrementing t so fnd matches t_no_dead's convention
            self.metrics.record_round(self.net, self.t,
                                      self._collect_family_metrics(),
                                      routing_tree=self._routing_tree)
            self.t += 1

        self.metrics.finalize()
```

Then add this helper method to `BaseAlgorithm` (after `_collect_family_metrics`):

```python
    def _charge_cluster_maintenance(self):
        """Build the cluster routing tree, stash it, and charge energy to every
        alive node along the CM->relay->CH->backbone->BS path (forward-only
        members, CH aggregation), tracking deaths. Returns the tree.
        """
        net = self.net
        tree = net.build_cluster_routing_tree()
        self._routing_tree = tree
        costs = net.compute_cluster_maintenance_costs(tree)
        for s in net.sensors:
            if not s.is_alive or s.id not in costs:
                continue
            cost = costs[s.id]
            if s.is_ch:
                s.c_ch = cost
            else:
                s.c_cm = cost
            s.e_res -= cost
            if s.e_res <= 0:
                self._track_death(s)
        return tree
```

- [ ] **Step 6: Run the cluster tests + the existing metrics tests**

Run: `conda run -n base python tests/test_cluster_routing.py`
Expected: prints the pass line, no assertion error.

Run: `conda run -n base python tests/test_metrics.py`
Expected: existing metrics tests still pass (the old `FakeNet.build_routing_tree` returns dicts without `delivers`; `record_round` uses `.get('delivers', True)` so they default to delivering).

- [ ] **Step 7: Commit**

```bash
git add model.py metrics.py algos/__init__.py tests/test_cluster_routing.py
git commit -m "feat: build-once-pass routing tree to metrics; avg_hop None on no-delivery"
```

---

## Task 4: Convert GT2, LEACH, GTFR, FL-LEACH-PSO, SCA-Lévy maintenance

These five share the identical maintenance body; each becomes a single call to
`self._charge_cluster_maintenance()`. GT2 also has a clusterless fallback to
stash.

**Files:** `algos/gt2.py`, `algos/leach.py`, `algos/gtfr.py`, `algos/fl_leach_pso.py`, `algos/sca_levy.py`

- [ ] **Step 1: GT2 `_maintenance`** (`algos/gt2.py:359-381`)

Replace the body of `_maintenance` after the docstring:

```python
        net = self.net

        routing_tree = net.build_routing_tree()
        costs = net.compute_maintenance_costs(routing_tree)

        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.is_ch:
                info = routing_tree.get(s.id)
                tx_dist = info['tx_dist'] if info else net.dist_to_bs(s)
                s.c_ch = (net.m_pkt_l * (net.e_elec + net.e_agg)
                          + net.calc_tx_cost(tx_dist, 'CH'))
                s.e_res -= s.c_ch
                if s.e_res <= 0:
                    self._track_death(s)
            elif s.id in costs:
                s.c_cm = costs[s.id]
                s.e_res -= s.c_cm
                if s.e_res <= 0:
                    self._track_death(s)
```

with:

```python
        self._charge_cluster_maintenance()
```

- [ ] **Step 2: GT2 `_maintenance_no_cluster`** (`algos/gt2.py:383-394`)

This clusterless fallback keeps the physical tree but must stash it. Replace its body after the docstring:

```python
        net = self.net
        routing_tree = net.build_routing_tree()
        costs = net.compute_maintenance_costs(routing_tree)
        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.id in costs:
                s.c_cm = costs[s.id]
                s.e_res -= s.c_cm
                if s.e_res <= 0:
```

with (add the stash line; keep the rest):

```python
        net = self.net
        routing_tree = net.build_routing_tree()
        self._routing_tree = routing_tree
        costs = net.compute_maintenance_costs(routing_tree)
        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.id in costs:
                s.c_cm = costs[s.id]
                s.e_res -= s.c_cm
                if s.e_res <= 0:
```

(Leave the `self._track_death(s)` line that follows unchanged.)

- [ ] **Step 3: LEACH `_steady_state`** (`algos/leach.py:243-267`)

Replace everything after the docstring (the `net = self.net` line through the final `self._track_death(s)`) with:

```python
        self._charge_cluster_maintenance()
```

- [ ] **Step 4: GTFR `_steady_state`** (`algos/gtfr.py:472-496`)

Replace everything after the docstring with:

```python
        self._charge_cluster_maintenance()
```

- [ ] **Step 5: FL-LEACH-PSO `_steady_state`** (`algos/fl_leach_pso.py:704-727`)

Replace everything after the docstring with:

```python
        self._charge_cluster_maintenance()
```

- [ ] **Step 6: SCA-Lévy maintenance** (`algos/sca_levy.py`, the method containing lines 488-508)

The method signature keeps its extra params (`layered_batches`, `ch_routes`).
Replace everything after the docstring (`net = self.net` through the final
`self._track_death(s)`) with:

```python
        self._charge_cluster_maintenance()
```

- [ ] **Step 7: Smoke-test all five for 3 rounds**

Create `tests/test_cluster_smoke.py`:

```python
"""Smoke test: clustering algos run a few rounds and drain energy.

Run:  conda run -n base python tests/test_cluster_smoke.py
"""
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_network, make_algo
from benchmark import _disable_plotting

CLUSTERING = ['GT2', 'LEACH', 'GTFR', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA']


def test_clustering_algos_run_and_drain():
    for algo in CLUSTERING:
        net = build_network(scenario='scenarios/gen/uniform_n200_s1.csv')
        e0 = sum(s.e_res for s in net.sensors)
        a = make_algo(algo, net, dict(max_rounds=3, plot_period=10 ** 9))
        _disable_plotting()
        a.run()
        e1 = sum(s.e_res for s in net.sensors)
        assert e1 < e0, f'{algo}: energy did not decrease'
        ts = a.metrics.time_series()
        assert len(ts['avg_hop']) >= 1, f'{algo}: no rounds recorded'
        print(f'{algo}: ok (drained {e0 - e1:.4f} J in {len(ts["rounds"])} rounds)')


if __name__ == '__main__':
    test_clustering_algos_run_and_drain()
    print('Task 4 smoke passed')
```

Run: `conda run -n base python tests/test_cluster_smoke.py`
Expected: each of GT2/LEACH/GTFR/FL-LEACH-PSO/SCA-LEVY prints `ok` (FC-CRA is converted in Task 5; it is included here and must also pass once Task 5 lands — for now it may use the not-yet-converted path and still run, so it is acceptable for it to print `ok` too).

- [ ] **Step 8: Commit**

```bash
git add algos/gt2.py algos/leach.py algos/gtfr.py algos/fl_leach_pso.py algos/sca_levy.py tests/test_cluster_smoke.py
git commit -m "feat: route GT2/LEACH/GTFR/FL-LEACH-PSO/SCA-Levy energy via cluster tree"
```

---

## Task 5: Convert FC-CRA and EE-TCM (with their extras)

**Files:** `algos/fc_cra.py`, `algos/ee_tcm.py`

- [ ] **Step 1: FC-CRA `_maintenance`** (`algos/fc_cra.py:461-484`)

FC-CRA flips `self._cluster_stable = False` when a CH dies. Replace everything
after the docstring with a cluster-tree version that preserves that:

```python
        net = self.net
        tree = net.build_cluster_routing_tree()
        self._routing_tree = tree
        costs = net.compute_cluster_maintenance_costs(tree)
        for s in net.sensors:
            if not s.is_alive or s.id not in costs:
                continue
            cost = costs[s.id]
            if s.is_ch:
                s.c_ch = cost
            else:
                s.c_cm = cost
            s.e_res -= cost
            if s.e_res <= 0:
                self._track_death(s)
                if s.is_ch:
                    self._cluster_stable = False
```

- [ ] **Step 2: EE-TCM `_maintenance`** (`algos/ee_tcm.py:458-488`)

EE-TCM applies CM compression and skips CMs that have not `_entered`. Keep the
compression setup; replace the deduction loop. Replace everything after the
docstring with:

```python
        net = self.net
        tree = net.build_cluster_routing_tree()
        self._routing_tree = tree
        costs = net.compute_cluster_maintenance_costs(tree)

        compression_enabled = self._compression_a > 1
        cm_compression = (1.0 / self._compression_a
                          if compression_enabled else 1.0)
        cm_overhead = self._compression_overhead if compression_enabled else 0.0

        for s in net.sensors:
            if not s.is_alive or s.id not in costs:
                continue
            if s.is_ch:
                s.c_ch = costs[s.id]
                s.e_res -= s.c_ch
                if s.e_res <= 0:
                    self._track_death(s)
            else:
                if not s._entered:
                    continue
                cost = costs[s.id] * cm_compression + cm_overhead
                s.c_cm = cost
                s.e_res -= cost
                if s.e_res <= 0:
                    self._track_death(s)
```

- [ ] **Step 3: Smoke-test FC-CRA and EE-TCM**

Run: `conda run -n base python tests/test_cluster_smoke.py`
Expected: FC-CRA prints `ok` (now on the cluster path).

Then EE-TCM directly:

Run:
```bash
conda run -n base python -c "
import os; os.environ['MPLBACKEND']='Agg'
from main import build_network, make_algo
from benchmark import _disable_plotting
net = build_network(scenario='scenarios/gen/uniform_n200_s1.csv')
e0 = sum(s.e_res for s in net.sensors)
a = make_algo('EE-TCM', net, dict(max_rounds=3, plot_period=10**9)); _disable_plotting(); a.run()
assert sum(s.e_res for s in net.sensors) < e0
print('EE-TCM ok')
"
```
Expected: `EE-TCM ok`

- [ ] **Step 4: Commit**

```bash
git add algos/fc_cra.py algos/ee_tcm.py
git commit -m "feat: route FC-CRA and EE-TCM energy via cluster tree (keeping their extras)"
```

---

## Task 6: Topology games stash `self._routing_tree`

The 4 topology games keep `build_routing_tree`/`compute_maintenance_costs`; they
only need to stash the tree so `record_round` uses the same one.

**Files:** `algos/dia_mia.py`, `algos/tcle.py`, `algos/eftcg.py`

- [ ] **Step 1: DIA-MIA** (`algos/dia_mia.py:348`)

Find:

```python
        routing_tree = net.build_routing_tree()
        costs = net.compute_maintenance_costs(routing_tree)
```

Replace with:

```python
        routing_tree = net.build_routing_tree()
        self._routing_tree = routing_tree
        costs = net.compute_maintenance_costs(routing_tree)
```

- [ ] **Step 2: TCLE** (`algos/tcle.py:418`)

Apply the identical change (insert `self._routing_tree = routing_tree` between
the `build_routing_tree()` and `compute_maintenance_costs(...)` lines).

- [ ] **Step 3: EFTCG** (`algos/eftcg.py:325`)

Apply the identical change (insert `self._routing_tree = routing_tree` between
the `build_routing_tree()` and `compute_maintenance_costs(...)` lines).

- [ ] **Step 4: Smoke-test the topology games**

Run:
```bash
conda run -n base python -c "
import os; os.environ['MPLBACKEND']='Agg'
from main import build_network, make_algo
from benchmark import _disable_plotting
for algo in ['DIA','MIA','TCLE','EFTCG-1','EFTCG-2']:
    net = build_network(scenario='scenarios/gen/uniform_n200_s1.csv')
    e0 = sum(s.e_res for s in net.sensors)
    a = make_algo(algo, net, dict(max_rounds=3, plot_period=10**9)); _disable_plotting(); a.run()
    assert sum(s.e_res for s in net.sensors) < e0
    assert a.metrics.time_series()['delivered'][0] >= 0
    print(algo,'ok')
"
```
Expected: each prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add algos/dia_mia.py algos/tcle.py algos/eftcg.py
git commit -m "feat: topology games stash routing tree for metric consistency"
```

---

## Task 7: Verify paper_tables None-handling (no code change expected)

`paper_tables.round_table` already filters `None` via
`if (v := value_at(ts, key, r)) is not None`, and `cell()` returns `--` when the
list is empty. So `avg_hop=None` rounds are excluded automatically.

**Files:** `paper_tables.py` (verify only)

- [ ] **Step 1: Confirm the filter exists**

Run: `grep -n "is not None" paper_tables.py`
Expected: shows the list-comprehension guard in `round_table`. No change needed.

- [ ] **Step 2: (Only if the guard is absent)** add it to `round_table`:

```python
            vals = [v for ts in series
                    if (v := value_at(ts, key, r)) is not None]
```

- [ ] **Step 3: No commit if nothing changed.** Otherwise:

```bash
git add paper_tables.py
git commit -m "fix: exclude no-delivery (None) avg_hop from paper tables"
```

---

## Task 8: Re-run the clustering sweep and regenerate outputs

Not a code task — a data-regeneration checklist. The 4 topology games' stored
runs stay valid; only the clustering algos must be re-swept.

- [ ] **Step 1: Delete the stale clustering-algo run files**

```bash
cd /home/quanh/Documents/workspace/github/gt-tc
rm -f results/runs/GT2_*.json results/runs/LEACH_*.json results/runs/GTFR_*.json \
      results/runs/FL-LEACH-PSO_*.json results/runs/SCA-LEVY_*.json results/runs/FC-CRA_*.json
```

- [ ] **Step 2: Re-run the sweep (resumable — skips the kept topology-game JSON)**

```bash
conda run -n base python benchmark.py
```
Expected: dispatches ~360 runs (6 clustering algos × 6 scenarios × 10 seeds); the DIA/MIA/TCLE/EFTCG JSONs are skipped. Writes `results/summary.csv`.

- [ ] **Step 3: Regenerate figures and the per-scenario CSV**

```bash
conda run -n base python plot_benchmark.py
```

- [ ] **Step 4: Regenerate the paper tables**

```bash
conda run -n base python paper_tables.py
```
Sanity-check: the clustering algos' `avg_hop` rows are now larger (true CM→CH→backbone path length) and `--` appears where no seed delivered.

- [ ] **Step 5: Refill the paper** (`~/Documents/workspace/github/GT2_paper/sec4_simulation.tex`) from the regenerated `paper_tables.py` output (avg_hop / avg_tx_power / epp tables) and add one methodology sentence: clustering protocols are charged along their CM→CH→backbone→BS data path; topology games over the shared physical graph. Recompile (`latexmk -pdf main.tex`) and confirm no overfull/errors.

- [ ] **Step 6: Commit the regenerated results**

```bash
cd /home/quanh/Documents/workspace/github/gt-tc
git add results/summary.csv results/summary_by_scenario.csv
git commit -m "data: re-sweep clustering algos under cluster-aware routing"
```

---

## Self-Review notes (for the executor)

- **Energy model invariant:** only the routing *topology* changed; `calc_tx_cost`,
  `e_elec`, `e_agg`, `eps_amp` are reused unchanged (Tasks 1-2).
- **Metric == energy:** guaranteed because each algo stashes the exact tree it
  charged on `self._routing_tree`, and `run()` passes that to `record_round`
  (Task 3). The `None` reset in `run()` makes a missing tree fall back safely.
- **Expected result shift:** clustering algos drain more per round (every alive
  node now transmits), so FND/HND/LND drop and `avg_hop` rises — this is the
  intended faithful model, not a regression.
