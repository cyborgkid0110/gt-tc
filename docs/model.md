# model.py Documentation

`model.py` provides two core classes — `Sensor` and `NetworkModel` — that together represent the physical layer of a Wireless Sensor Network (WSN). These classes encapsulate the node state, network topology, and energy calculations that are **shared across all algorithms** (GT2, LEACH, EGCR, etc.).

---

## Class: `Sensor`

A `Sensor` instance represents a single sensor node deployed in the WSN. It holds all per-node state that evolves during the simulation.

### Constructor

```python
Sensor(id, x, y, e0, power, Vpre)
```

| Parameter | Description |
|-----------|-------------|
| `id` | Unique integer index matching the node's row/column in the global adjacency matrix. |
| `x`, `y` | Coordinates in the 2-D deployment area (centred at the origin). |
| `e0` | Initial energy budget (Joules). Also stored for later reference by the energy-cost function. |
| `power` | Current transmission power level (Watts). Determines the communication range. |
| `Vpre` | Supply voltage of this node's hardware, drawn at random during generation. Used in sensing/processing energy calculations. |

### Attributes

#### Physical state

| Attribute | Type | Description |
|-----------|------|-------------|
| `e_res` | `float` | Residual energy. Decremented each round during the maintenance phase. A node is considered dead when `e_res <= 0`. |
| `rc` | `float` | Communication range (metres). Derived from `power` via the Friis equation; kept in sync by `NetworkModel.update_comm_range()`. |

#### Neighbour / cluster bookkeeping

| Attribute | Type | Description |
|-----------|------|-------------|
| `neighbors` | `list[Sensor]` | Direct (1-hop) neighbours within the same cluster. Rebuilt every round. |
| `ch_neighbors` | `list[Sensor]` | Other cluster heads visible to this node (only populated when `is_ch == True`). Used for inter-cluster routing. |
| `is_ch` | `bool` | Whether this node is currently acting as a Cluster Head. |
| `ch_belong` | `Sensor \| None` | Reference to the CH this node is affiliated with. `None` for unaffiliated or isolated nodes. |

#### Clustering game state

| Attribute | Type | Description |
|-----------|------|-------------|
| `p0` | `float \| None` | Base CH election probability computed from the cost difference `(C_CH - C_CM)` and the payoff. |
| `p_ch` | `float \| None` | Actual probability used in the random draw. Set only when the node enters the candidate pool. |
| `c_ch` | `float` | Energy cost if the node acts as CH this round. |
| `c_cm` | `float` | Energy cost if the node acts as CM this round. |

#### Power-control game state

| Attribute | Type | Description |
|-----------|------|-------------|
| `util` | `float \| None` | Current utility value in the power-control game. `None` until first evaluated. |
| `local_net` | `dict \| None` | Local sub-graph `{'vertices': [Sensor, ...], 'edges': np.ndarray}` representing the k-hop neighbourhood. Built lazily during the adaptation phase. |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `pos` | `tuple[float, float]` | `(x, y)` coordinate tuple. Used as the key in legacy `node_dict` format and for position lookups. |
| `is_alive` | `bool` | `True` when `e_res > 0`. Dead nodes are excluded from all game phases. |

### Methods

#### `distance_to(other: Sensor) -> float`

Euclidean distance between this node and `other`. Used everywhere: neighbour discovery, cluster joining, power-reduction checks.

#### `add_neighbor(other: Sensor)`

Appends `other` to `self.neighbors` if not already present. Does **not** touch the global adjacency matrix — use `NetworkModel.connect()` when you need both edge and neighbour updated atomically.

#### `remove_neighbor(other: Sensor)`

Removes `other` from `self.neighbors`. Same caveat as `add_neighbor` regarding the adjacency matrix.

#### `reset_round()`

Clears per-round state to prepare for a new simulation round. Resets:
- `neighbors` (emptied — rebuilt during neighbour discovery)
- `ch_belong` (cleared — reassigned during cluster formation)
- `util`, `local_net` (cleared — recomputed during adaptation)
- `p_ch` (cleared — recomputed during the clustering game)

Note: `is_ch` is intentionally **not** reset here. It is carried into the next round so that the clustering game can detect and skip previous-round CHs before clearing the flag.

---

## Class: `NetworkModel`

`NetworkModel` owns the collection of sensors, the global adjacency matrix, and all physics/energy computations. It is designed to be **algorithm-agnostic**: any topology-control algorithm receives a `NetworkModel` instance and calls its methods without reimplementing the energy model.

### Constructor

```python
NetworkModel(sensors, area, **params)
```

| Parameter | Description |
|-----------|-------------|
| `sensors` | `list[Sensor]` — the pre-created node population. |
| `area` | Side length of the square deployment region. Used only by `check_potential_connectivity()` and plotting. |

Keyword parameters (all optional, with defaults matching the paper's simulation setup):

| Key | Default | Description |
|-----|---------|-------------|
| `pth` | `7e-10` | Signal capture threshold P_th (Watts). |
| `wave` | `0.1224` | Carrier wavelength lambda (metres). |
| `p_min` | `0.01` | Minimum allowed transmission power. |
| `p_max` | `0.08` | Maximum allowed transmission power. |
| `p_step` | `0.0001` | Power decrement step for the iterative power-control game. |
| `hop_max` | `3` | Maximum hop count for k-hop local-network construction. |
| `e_elec` | `50e-9` | Electronics energy per bit (Joules/bit). |
| `e_agg` | `5e-9` | Data aggregation energy per bit. |
| `m_pkt_s` | `20` | Small packet size in bits (CM data). |
| `m_pkt_l` | `1000` | Large packet size in bits (CH aggregated data). |
| `alpha` | `1.5` | Weight for the energy-balance benefit `f_e`. |
| `beta` | `0.1` | Weight for the connectivity benefit `f_pr`. |
| `mu` | `0.01` | Scaling factor for the integral energy-cost function `c_i`. |

After construction the communication range of every sensor is initialised from its current power.

---

### Communication Range

#### `calc_comm_range(power) -> float`

Derives the maximum transmission range from the Friis free-space equation:

```
R_tx = (lambda / 4*pi) * sqrt(P_t / P_th)
```

Antenna gains are absorbed into the constants. This is the single source of truth for the power-to-range mapping; every other method that needs a range calls this.

#### `calc_rx_power(p_tx, d) -> float`

Inverse of the range calculation: returns the received power at distance `d` given transmit power `p_tx`. Can be used for link-budget checks.

#### `update_comm_range(sensor)`

Convenience wrapper: recalculates `sensor.rc` from `sensor.power`. Call this after any direct modification of a sensor's power.

---

### Energy Model

These methods implement the energy consumption formulas from the paper. They are pure computations with no side effects on sensor state.

#### `calc_tx_cost(d, role, layer_depth=1) -> float`

Transmission energy for sending one packet over distance `d`. The packet size depends on the `role`:

- **CM**: uses `m_pkt_s` (small sensing packet).
- **CH**: uses `m_pkt_l` (aggregated packet forwarded to sink).

The `layer_depth` multiplier models multi-hop relay costs — nodes deeper in the cluster tree forward data through more intermediate hops.

The amplifier term uses the Friis-derived dissipation factor `(4*pi/lambda)^2 * P_th * t_tx`, making the cost proportional to `d^2`.

#### `calc_node_cost(sensor, role, clustering, layer_depth=1) -> float`

Total energy cost for one round, combining sensing, processing, and transmission.

The `clustering` flag controls how the reference distance is chosen:
- `clustering=True` (used during CH election): distance to the sink at origin `sqrt(x^2 + y^2)`. This reflects the expected forwarding cost when deciding whether to become a CH.
- `clustering=False` (used during maintenance): the sensor's current communication range `rc`. This reflects actual transmission cost.

For **CM** role, sensing and processing costs are added. Sensing current is drawn from a uniform random range to model hardware variability. For **CH** role, reception and aggregation costs replace sensing/processing.

---

### Utility Functions (Power-Control Game)

These methods compute the three components of the utility function used in the intra-cluster transmission power control game.

#### `calc_energy_cost(sensor, tx_power) -> float`

Implements the integral-based cost `c_i(p_i)`:

```
c_i = (1/mu) * integral from (E0 - E_res) to (E0 - E_res + p_i * T) of exp(x/10) dx
```

The exponential integrand makes cost grow sharply as a node's cumulative energy consumption increases. This discourages nearly-depleted nodes from choosing high power, promoting network lifetime.

#### `calc_connectivity_benefit(sensor, hop) -> float`

Computes `f_pr(i) = beta * |V'_i|`, where `|V'_i|` is the number of nodes in the k-hop neighbourhood. More neighbours means more value to the network topology, rewarding nodes that maintain connectivity.

#### `calc_energy_balance_benefit(local_net) -> float`

Computes the variance of residual energies across the local network:

```
f_e = alpha * sum((E_r(j) - E_avg)^2) / |V'|
```

This term is **subtracted** in the utility function, penalising topologies where energy levels are unbalanced. It drives the network toward uniform energy depletion.

#### `calc_utility(sensor, power) -> float`

Combines the three components into the final utility, assuming the node is connected (`f_i = 1`):

```
u_i = f_pr(i) - f_e(i) - c_i(p_i)
```

When a node loses connectivity, the calling code assigns a large negative utility externally (not handled inside this method).

---

### Topology Operations

#### `get_k_hop_vertices(sensor, hop) -> list[Sensor]`

Recursively collects all sensors reachable within `hop` hops by following the `neighbors` list. CHs act as barriers — the traversal stops at any CH to keep the neighbourhood within cluster boundaries.

Returns a deduplicated list including the sensor itself.

#### `get_local_graph(sensor, hop) -> dict`

Builds the local sub-graph by:
1. Calling `get_k_hop_vertices` to get the vertex set.
2. Constructing a local adjacency matrix by checking which pairs are mutual neighbours.

Returns `{'vertices': [Sensor, ...], 'edges': np.ndarray}`.

The local graph is used for connectivity verification and energy-balance calculations during the power-control game.

#### `check_local_connectivity(local_net, start_sensor) -> bool`

Runs a DFS from `start_sensor` over the local adjacency matrix. Returns `True` only if every vertex in the local network is reachable. The power-control game uses this to reject power reductions that would disconnect the local topology.

#### `update_edges_from_local(local_net)`

Writes the local adjacency matrix back into the global `self.edges`. Called after a power reduction is accepted to propagate topology changes to the global graph.

#### `check_potential_connectivity() -> bool`

Pre-simulation sanity check: verifies that every node can reach at least one other node when all nodes transmit at `p_max`. If this fails, the node deployment is too sparse for the given power budget.

---

### Edge / Neighbour Management

#### `connect(src, dst)`

Atomically sets `edges[src.id, dst.id] = 1` **and** adds `dst` to `src.neighbors`. Use this instead of manipulating edges and neighbours separately to avoid inconsistency.

#### `disconnect(src, dst)`

Atomically clears the edge **and** removes the neighbour entry.

#### `discover_neighbors()`

The broadcast / ADV phase: iterates over all alive sensor pairs and calls `connect()` for every pair where the distance is within the source's communication range. This populates both the adjacency matrix and every sensor's neighbour list in one pass.

#### `reset_round()`

Zeros the adjacency matrix and calls `sensor.reset_round()` on every node. Called once at the start of each simulation round.

#### `create_cluster_subgraph() -> np.ndarray`

Returns a **copy** of the global adjacency matrix with two kinds of edges removed:
1. CH-to-CM edges where the distance exceeds the default CM communication range (`p_max / 4`). These are long-range links that only exist because CHs transmit at `p_max`.
2. CH-to-CH edges (stored in `ch_neighbors`).

The result is a "CM-only" view of each cluster, suitable for BFS layer assignment in `graph.divide_network_by_clusters()`.

---

### Compatibility Helpers

These methods convert the object-oriented representation back into the legacy dict-based format expected by `graph.py` and `plot.py`.

#### `to_network_dict(edges=None) -> dict`

Returns `{'vertices': [(x,y), ...], 'edges': np.ndarray}`. If `edges` is provided (e.g., from `create_cluster_subgraph()`), that matrix is used instead of the global one.

#### `to_node_dict() -> dict`

Returns a dict keyed by position tuples `(x, y)` with the same attribute names used by the legacy code (`'CH'`, `'CH_belong'`, `'neighbors'`, etc.). Sensor object references are converted to position tuples so that `graph.py`'s lookup-by-position logic continues to work.

#### `sensor_by_pos(pos) -> Sensor | None`

Reverse lookup: given a position tuple (typically returned from `graph.py`'s layer assignment), returns the corresponding `Sensor` object. Uses an internal hash map built at construction time for O(1) access.
