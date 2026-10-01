# EE-TCM Topology Control Game: Entrance Sub-Game Reimplementation

**Date:** 2026-05-30
**Scope:** Reimplement Phase 2 of `algos/ee_tcm.py` to include the paper's entrance sub-game (§3.5) before the power-control game.

---

## Problem

The current EE-TCM topology control game (`_adapt_topology`) is a continuous power-reduction game identical to EFTCG. It skips the paper's entrance/harvesting sub-game entirely — every node always participates. The paper defines a two-layer game:

1. **Entrance sub-game:** binary decision `x_i(j) in {0, 1}` (enter or stay out) with mixed-strategy NE.
2. **Power-control game:** continuous `p_i` reduction for entering nodes using `u_i = f_k * (alpha_i * power_saving + beta_i * avg_neighbor_energy)`.

The current implementation only has layer 2.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Harvested energy `f_i` | `f_i = 0` (no energy consumption for staying out) | Project's NetworkModel has no harvesting mechanism. Staying out = cost avoidance. |
| Game sequence | Sequential: entrance then power-control | Paper describes two distinct phases; entering nodes feed into the power game. |
| `g_i(j)` (residual energy) | `s.e_res` (raw Joules) | Direct mapping to paper's definition. |
| `C_i(j)` (transmission cost) | `net.calc_tx_cost(s.distance_to(ch), 'CM')` | Uses project's shared energy model; tx cost to the node's CH. |
| NE computation | Closed-form analytical solution | O(n) per cluster, faithful to paper's mixed-strategy NE. |

---

## Round Flow (Updated)

```
Phase 1: Clustering (unchanged)
  -> _ch_election, _cluster_formation, _filter_neighbours,
     _connect_unaffiliated, _cleanup_cross_cluster_edges

Phase 2: Topology Control Game (reimplemented)
  -> Step 2a: _entrance_game()          <- NEW
  -> Step 2b: _adapt_topology()         <- MODIFIED

Phase 3: Maintenance (modified)
  -> Entering CMs: full energy cost
  -> Staying-out CMs: zero energy cost
  -> CHs: always pay (unchanged)
```

---

## Entrance Game (`_entrance_game`)

Operates per-cluster. CHs always enter; the game is among CMs only.

### Step 1: Compute game parameters

For each CM `j` in cluster `i`:
- `g_j = s.e_res`
- `C_j = net.calc_tx_cost(s.distance_to(ch), 'CM')`
- If `C_j >= g_j`: node is **forced-out** (`P_j = 0`), excluded from NE computation

### Step 2: Solve mixed NE

For `n` eligible CMs (those with `g_j > C_j`):

- `n == 0`: no one can enter; CH operates alone
- `n == 1`: single node always enters (`P_j = 1`)
- `n >= 2`: closed-form NE from the indifference condition `prod_{k != j}(1 - P_k) = C_j / g_j`:

```
R = (prod_j C_j/g_j) ^ (1/(n-1))
P_j = 1 - R * g_j / C_j
P_j = clamp(P_j, 0.0, 1.0)
```

**Derivation:** At mixed NE, each node is indifferent between entering and staying out. The paper's expected utility (with `f_i = 0`):

```
E[U_j] = P_j * (g_j - C_j) + (1 - P_j) * g_j * (1 - prod_{k != j}(1 - P_k))
```

Setting `dE[U_j]/dP_j = 0`:

```
g_j - C_j = g_j * (1 - prod_{k != j}(1 - P_k))
=> prod_{k != j}(1 - P_k) = C_j / g_j
```

Letting `R = prod_all(1 - P_k)`, then `R / (1 - P_j) = C_j / g_j` for all `j`, giving `(1 - P_j) = R * g_j / C_j`. Substituting back: `R = R^n * prod(g_j/C_j)`, solving for R yields `R = (prod(C_j/g_j))^{1/(n-1)}`.

### Step 3: Stochastic realization

Each eligible CM draws `random.random() < P_j` to decide. Mark entering CMs with `s._entered = True`.

### Step 4: Scope control (no edge mutation needed)

Staying-out CMs are **not disconnected** from the graph. Instead:
- `_adapt_topology` filters the `members` list to entering CMs + CH. Since `_cluster_digraph` only adds edges between IDs in its `members` set, staying-out nodes are naturally excluded from the cluster digraph.
- `_neighbor_energy` is modified to accept an `active_ids` set and only count neighbors whose ID is in that set, so staying-out nodes don't affect the avg neighbor energy of entering nodes.
- `_maintenance` checks `s._entered` and skips staying-out CMs in the CM energy loop.

This avoids mutating `net.edges` or neighbor lists, keeping changes localized.

### Edge cases

- All CMs stay out: CH operates alone, no intra-cluster data flow that round.
- Single CM: always enters (no game).
- `C_j >= g_j`: forced out, too expensive to transmit.

---

## Power-Control Game Modifications (`_adapt_topology`)

### Changes

- Filter `members` list per cluster to **entering CMs + CH only**. Staying-out nodes are excluded from `_cluster_digraph`.
- `_f_k` evaluates connectivity over the active-participant subgraph only.
- `_neighbor_energy` accepts an `active_ids: set[int]` parameter and only counts neighbors in that set, so staying-out nodes don't influence the utility of entering nodes.

### Unchanged

- Sequential better-response power-reduction loop.
- Fast/slow-path split (no links lost -> auto-accept if alpha_i > 0).
- Utility function `u_i = f_k * (alpha_i * power_saving + beta_i * avg_neighbor_energy)`.
- CH stays at `p_max`.

---

## Maintenance Modifications (`_maintenance`)

| Node type | Energy cost |
|-----------|------------|
| Entering CMs | Full `calc_node_cost(s, 'CM', ...)` with optional compression |
| Staying-out CMs | Zero |
| CHs | Unchanged (aggregation + CH->BS tx) |
| Unaffiliated nodes | Unchanged |

The `_entered` flag is reset at the start of each round alongside the existing per-round resets.

---

## Files Modified

| File | Change |
|------|--------|
| `algos/ee_tcm.py` | Add `_entrance_game()` method; modify `_adapt_topology()` to filter by entered nodes; modify `_maintenance()` to skip staying-out CMs; reset `_entered` flag in `_run_round()`. |

No config changes needed — no new parameters introduced.

---

## Constraints

- Uses the project's shared energy model (`calc_tx_cost`, `calc_node_cost`). No custom energy formulas.
- Directed adjacency matrix convention: `_cluster_digraph` and `_f_k` use strong connectivity on DiGraph.
- No `deepcopy` on Sensor objects.
