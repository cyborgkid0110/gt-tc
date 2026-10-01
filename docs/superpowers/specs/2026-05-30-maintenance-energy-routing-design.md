# Maintenance Phase Energy Model: Routing-Based Per-Hop TX Cost

**Date:** 2026-05-30
**Scope:** Replace the flat `sensor.rc`-based TX distance in all algorithms' maintenance phases with actual next-hop distances derived from shortest-path routing to BS, including relay forwarding costs.

---

## Problem

`calc_node_cost(s, 'CM', clustering=False)` uses `d = sensor.rc` (the node's maximum communication range) as the TX distance. In a multi-hop network, a CM transmits to its **next-hop neighbor**, not at its max range. This overcharges every node by using a worst-case distance instead of the actual hop distance.

Additionally, relay nodes (intermediate hops on the path to BS) are not charged for forwarding traffic from downstream nodes. In reality, relay nodes pay RX + TX for every packet they forward.

This affects all 10 algorithms: GT2, LEACH, GTFR, DIA/MIA, TCLE, EFTCG, FL-LEACH-PSO, SCA-Levy, FC-CRA, EE-TCM.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| TX distance | Actual distance to next-hop on shortest path to BS | Physically correct; nodes transmit to their relay neighbor, not at max range |
| Relay cost | Own data + forwarding (RX + TX per descendant) | Relay nodes must receive and retransmit data from downstream nodes |
| Implementation | Unified helper in `NetworkModel` | DRY — one routing + energy method serves all algorithms |
| Routing algorithm | Reverse BFS from BS through directed edges | Simple, correct for directed graphs, O(V+E) |
| Nodes without a route | Pay sensing + processing only, no TX | Cannot reach BS, no transmission |
| CH costs | Handled separately per algorithm (unchanged) | CHs perform aggregation and use large packets; semantics vary by algorithm |
| Game-theory phases | Untouched | `calc_node_cost(clustering=True)` in GT2/GTFR is for cost estimation, not maintenance |

---

## New Methods in `NetworkModel`

### `build_routing_tree() -> dict[int, RoutingInfo]`

Builds a shortest-path (fewest hops) tree from all alive nodes to BS (origin at 0,0).

**Algorithm:**
1. Compute each alive node's distance to BS: `d_bs = hypot(s.x, s.y)`
2. Identify **gateway nodes**: alive nodes where `d_bs <= s.rc` (can reach BS directly)
3. Build a **reverse directed graph**: for each edge `A -> B` in `self.edges`, add `B -> A` in the reverse graph
4. Multi-source BFS from all gateway nodes through the reversed graph
5. For each discovered node, record:
   - `parent_id: int | None` — next-hop sensor ID toward BS (None for gateways)
   - `tx_dist: float` — actual euclidean distance to next-hop (or to BS for gateways)
   - `depth: int` — hop count to BS (0 for gateways)
6. After BFS, compute `num_descendants: int` for each node by walking the tree bottom-up (sum of subtree sizes)

**Returns:** `dict[int, RoutingInfo]` keyed by sensor ID. Nodes not in the dict have no route to BS.

**Why reverse BFS:** In the directed graph, an edge `A -> B` means A can transmit to B. For routing **to** BS, we need paths where data flows from any node toward BS. Reversing edges and BFS-ing from BS discovers all nodes that can reach BS via directed multi-hop paths.

### `compute_maintenance_costs(routing_tree) -> dict[int, float]`

Computes per-node energy cost using the routing tree.

For each node `s` with a valid route:

```
sense    = Vpre * i_sense * 8
process  = Vpre * 8 * i_sense / 4
tx_own   = m_pkt_s * (e_elec + eps_amp * tx_dist^gamma)
rx_fwd   = m_pkt_s * e_elec * num_descendants
tx_fwd   = m_pkt_s * (e_elec + eps_amp * tx_dist^gamma) * num_descendants

cost = sense + process + tx_own + rx_fwd + tx_fwd
     = sense + process + (1 + num_descendants) * TX(tx_dist) + num_descendants * RX
```

For nodes without a route: `cost = sense + process` (no TX/RX).

Returns `dict[int, float]` keyed by sensor ID.

---

## Algorithm Integration

### Clustering algorithms (GT2, LEACH, GTFR, EE-TCM, FL-LEACH-PSO, SCA-Levy, FC-CRA)

**Current pattern:**
```python
mod_edges = net.create_cluster_subgraph()
G = graph.build_graph(...)
layered_batches = graph.divide_network_by_clusters(G, node_dict)
for ch_pos, layers in layered_batches.items():
    # charge CH: c_ch = ...
    for layer in layers:
        for node in layer:
            cost = net.calc_node_cost(s, 'CM', clustering=False, layer_depth=...)
            s.e_res -= cost
```

**New pattern:**
```python
routing_tree = net.build_routing_tree()
costs = net.compute_maintenance_costs(routing_tree)
for s in net.sensors:
    if not s.is_alive:
        continue
    if s.is_ch:
        # CH: aggregation + RX + TX to BS (algorithm-specific, kept as-is)
        ...
    elif s.id in costs:
        s.c_cm = costs[s.id]
        s.e_res -= s.c_cm
        if s.e_res <= 0:
            self._track_death(s)
```

**CH costs remain separate** — each algorithm handles CH aggregation, RX from members, and TX to BS (direct or multi-hop) in its own way. The unified method only handles CM/relay costs.

### Non-clustering algorithms (DIA/MIA, TCLE, EFTCG)

**Current pattern:**
```python
for s in net.sensors:
    if not s.is_alive:
        continue
    cost = net.calc_node_cost(s, 'CM', clustering=False)
    s.c_cm = cost
    s.e_res -= cost
```

**New pattern:**
```python
routing_tree = net.build_routing_tree()
costs = net.compute_maintenance_costs(routing_tree)
for s in net.sensors:
    if not s.is_alive:
        continue
    if s.id in costs:
        s.c_cm = costs[s.id]
        s.e_res -= s.c_cm
        if s.e_res <= 0:
            self._track_death(s)
```

---

## What Gets Removed

- `calc_node_cost(s, role, clustering=False)` calls in maintenance phases (all algorithms)
- `create_cluster_subgraph()` / `divide_network_by_clusters()` calls in maintenance (clustering algorithms) — these may still be used elsewhere (plotting, etc.)
- The `layer_depth` multiplier pattern in maintenance

## What Stays Unchanged

- CH election, cluster formation, topology adaptation — all untouched
- CH-specific energy (aggregation, CH->BS TX) — per-algorithm, kept as-is
- `calc_node_cost(s, role, clustering=True)` in GT2/GTFR game-theory phases
- `calc_tx_cost()` and `calc_comm_range()` — used by the new methods internally
- `graph.py` — not modified; layered-batch functions remain available

---

## Files Modified

| File | Change |
|------|--------|
| `model.py` | Add `build_routing_tree()`, `compute_maintenance_costs()` |
| `algos/gt2.py` | Replace maintenance loop with unified method |
| `algos/leach.py` | Replace maintenance loop with unified method |
| `algos/gtfr.py` | Replace maintenance loop with unified method |
| `algos/dia_mia.py` | Replace maintenance loop with unified method |
| `algos/tcle.py` | Replace maintenance loop with unified method |
| `algos/eftcg.py` | Replace maintenance loop with unified method |
| `algos/fl_leach_pso.py` | Replace maintenance loop with unified method |
| `algos/sca_levy.py` | Replace maintenance loop with unified method |
| `algos/fc_cra.py` | Replace maintenance loop with unified method |
| `algos/ee_tcm.py` | Replace maintenance loop with unified method |

---

## Verification

Smoke-test each algorithm (3 rounds, no plots) to confirm:
1. No crashes
2. Energy is deducted each round (nodes eventually die)
3. Nodes closer to BS pay less TX energy than distant nodes
