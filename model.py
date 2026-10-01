import math
import random
import numpy as np
from scipy import integrate


class Sensor:
    """Represents a sensor node in the WSN."""

    def __init__(self, id, x, y, e0, power, Vpre):
        self.id = id
        self.x = x
        self.y = y
        self.e0 = e0
        self.e_res = e0
        self.Vpre = Vpre
        self.power = power
        self.rc = 0.0

        self.neighbors = []
        self.ch_neighbors = []

        self.p0 = None
        self.p_ch = None
        self.is_ch = False
        self.ch_belong = None       # reference to CH Sensor

        self.c_ch = 0.0
        self.c_cm = 0.0

        self.util = None
        self.local_net = None       # {'vertices': [Sensor, ...], 'edges': np.array}

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def is_alive(self):
        return self.e_res > 0

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def add_neighbor(self, other):
        if other not in self.neighbors:
            self.neighbors.append(other)

    def remove_neighbor(self, other):
        if other in self.neighbors:
            self.neighbors.remove(other)

    def reset_round(self):
        """Reset per-round clustering state (CH status preserved for next round)."""
        self.neighbors = []
        self.ch_belong = None
        self.util = None
        self.local_net = None
        self.p_ch = None

    def __repr__(self):
        return f"Sensor(id={self.id}, pos=({self.x:.1f}, {self.y:.1f}))"


class NetworkModel:
    """Manages WSN topology and energy calculations."""

    def __init__(self, sensors, area, **params):
        self.sensors = sensors
        self.num_nodes = len(sensors)
        self.area = area
        bs_pos = params.get('bs_pos', (0.0, 0.0))
        self.bs_x = float(bs_pos[0])
        self.bs_y = float(bs_pos[1])
        self.edges = np.zeros((self.num_nodes, self.num_nodes), dtype=int)

        self._pos_map = {s.pos: s for s in sensors}

        # Radio parameters — link-budget model (Tudose et al. Eq. 7–8)
        self.snr = params.get('snr', 10)
        self.nf_rx = params.get('nf_rx', 6.31)
        self.n0 = params.get('n0', 3.98e-21)
        self.bw = params.get('bw', 3e6)
        self.wave = params.get('wave', 0.125)
        self.gamma = params.get('gamma', 2.0)
        self.g_tx = params.get('g_tx', 1.0)
        self.g_rx = params.get('g_rx', 1.0)
        self.eta = params.get('eta', 0.30)
        self.r_bit = params.get('r_bit', 250e3)

        self.p_th = self.snr * self.nf_rx * self.n0 * self.bw
        self.eps_amp = (self.p_th * (4 * math.pi / self.wave) ** self.gamma
                        / (self.g_tx * self.g_rx * self.eta * self.r_bit))

        self.p_min = params.get('p_min', 0.01)
        self.p_max = params.get('p_max', 0.08)
        self.p_step = params.get('p_step', 0.0001)

        # Energy parameters
        self.e_elec = params.get('e_elec', 50e-9)
        self.e_agg = params.get('e_agg', 5e-9)
        self.m_pkt_s = params.get('data_payload', 32)            # data packet bits
        self.m_pkt_l = params.get('agg_payload', 72)             # aggregated packet bits
        self.sensor_sample_bits = params.get('sensor_sample_bits', 16)
        # deterministic sensing current (A) [1e-8, 5e-7]
        self.i_sense = params.get('i_sense', 2.55e-7)

        # Game 2 parameters
        self.alpha = params.get('alpha', 1.5)
        self.beta = params.get('beta', 0.1)
        self.mu = params.get('mu', 0.01)

        for s in self.sensors:
            s.rc = self.calc_comm_range(s.power)

    def sensor_by_pos(self, pos):
        """Look up a sensor by its position tuple."""
        return self._pos_map.get(pos)

    def dist_to_bs(self, sensor):
        """Euclidean distance from a sensor to the base station."""
        return math.hypot(sensor.x - self.bs_x, sensor.y - self.bs_y)

    # ------------------------------------------------------------------ #
    #  Communication range (Friis free-space model)                       #
    # ------------------------------------------------------------------ #

    def calc_comm_range(self, power):
        """Maximum single-hop range at given transmit power (d_max formula)."""
        return (power * self.g_tx * self.g_rx * self.eta
                / (self.p_th * (4 * math.pi / self.wave) ** self.gamma)
                ) ** (1 / self.gamma)

    def calc_power_for_range(self, rc):
        """Inverse of ``calc_comm_range``: transmit power for single-hop range rc."""
        return (rc ** self.gamma) * self.p_th \
            * (4 * math.pi / self.wave) ** self.gamma \
            / (self.g_tx * self.g_rx * self.eta)

    def update_comm_range(self, sensor):
        sensor.rc = self.calc_comm_range(sensor.power)

    # ------------------------------------------------------------------ #
    #  Energy model                                                       #
    # ------------------------------------------------------------------ #

    def calc_tx_cost(self, d, role, layer_depth=1):
        """Transmission energy: E_TX = n * (E_elec + eps_amp * d^gamma)."""
        m_bit = self.m_pkt_s if role == 'CM' else self.m_pkt_l
        return m_bit * layer_depth * (
            self.e_elec + self.eps_amp * d ** self.gamma
        )

    def calc_node_cost(self, sensor, role, clustering, layer_depth=1):
        """Total energy cost for a sensor in a given role.

        When *clustering* is True the distance is measured to the sink
        (origin).  Otherwise the sensor's current communication range
        is used as the reference distance.

        **Main Idea:** Sensing and processing costs are 
        applied to both CH and CM roles
        * Every node samples and processes its own data
        * Including baseline costs in both roles cancels 
        them out in the $c_{ch} - c_{cm}$ comparison.
        * Without this cancellation, the CM role carried 
        an extra sensing burden, making $c_{cm} > c_{ch}$ 
        almost always and breaking the game by forcing a 
        100% volunteer probability ($p_0 = 1$).
        * The actual energy drain path is unaffected; a 
        CH's functional cost remains receiving, aggregating, 
        and transmitting data.
        """
        if clustering:
            d = self.dist_to_bs(sensor)
        else:
            d = sensor.rc

        c_tx = self.calc_tx_cost(d, role, layer_depth)

        m_bit = self.sensor_sample_bits
        c_sense = sensor.Vpre * self.i_sense * m_bit
        c_process = sensor.Vpre * m_bit * self.i_sense / 4

        if role == 'CM':
            return c_sense + c_process + c_tx
        else:   # CH
            c_rx = self.m_pkt_l * self.e_elec
            c_agg = self.m_pkt_l * self.e_agg
            cost = c_rx + c_agg + c_tx
            if clustering:
                cost += c_sense + c_process
            return cost

    # ------------------------------------------------------------------ #
    #  Routing-based maintenance energy model                             #
    # ------------------------------------------------------------------ #

    def build_routing_tree(self):
        """Build shortest-path (fewest hops) tree from all alive nodes to BS.

        Returns dict[int, dict] keyed by sensor ID with:
          parent_id: int | None  (next-hop toward BS; None for gateways)
          tx_dist: float         (distance to next-hop or to BS)
          depth: int             (hop count to BS)
          num_descendants: int   (nodes in subtree, for relay cost)
        """
        from collections import deque

        gateways = []
        for s in self.sensors:
            if not s.is_alive:
                continue
            if self.dist_to_bs(s) <= s.rc:
                gateways.append(s.id)

        rev_adj = [[] for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            if not self.sensors[i].is_alive:
                continue
            for j in range(self.num_nodes):
                if i != j and self.sensors[j].is_alive and self.edges[i, j] == 1:
                    rev_adj[j].append(i)

        parent = {}
        depth = {}
        queue = deque()
        for gid in gateways:
            parent[gid] = None
            depth[gid] = 0
            queue.append(gid)

        while queue:
            curr = queue.popleft()
            for nb_id in rev_adj[curr]:
                if nb_id not in parent:
                    parent[nb_id] = curr
                    depth[nb_id] = depth[curr] + 1
                    queue.append(nb_id)

        descendants = {nid: 0 for nid in parent}
        for nid in sorted(parent, key=lambda x: depth[x], reverse=True):
            p = parent[nid]
            if p is not None:
                descendants[p] += 1 + descendants[nid]

        tree = {}
        for nid in parent:
            s = self.sensors[nid]
            pid = parent[nid]
            if pid is None:
                tx_d = self.dist_to_bs(s)
            else:
                tx_d = s.distance_to(self.sensors[pid])
            tree[nid] = {
                'parent_id': pid,
                'tx_dist': tx_d,
                'depth': depth[nid],
                'num_descendants': descendants[nid],
                'delivers': True,
            }
        return tree

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

    def compute_maintenance_costs(self, routing_tree):
        """Per-node energy cost based on routing tree (CM role only).

        Returns dict[int, float] keyed by sensor ID.
        Nodes without a route pay sensing + processing only.
        """
        costs = {}
        for s in self.sensors:
            if not s.is_alive:
                continue

            m_bit = self.sensor_sample_bits
            i_sense = random.uniform(1e-8, 5e-7)
            c_sense = s.Vpre * i_sense * m_bit
            c_process = s.Vpre * m_bit * i_sense / 4

            if s.id not in routing_tree:
                costs[s.id] = c_sense + c_process
                continue

            info = routing_tree[s.id]
            tx = self.m_pkt_s * (self.e_elec
                                 + self.eps_amp * info['tx_dist'] ** self.gamma)
            rx = self.m_pkt_s * self.e_elec
            nd = info['num_descendants']

            costs[s.id] = c_sense + c_process + (1 + nd) * tx + nd * rx

        return costs

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

    # ------------------------------------------------------------------ #
    #  Utility functions (power-control game)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _energy_cost_integrand(x):
        return np.exp(x / 10)

    def calc_energy_cost(self, sensor, tx_power):
        """c_i(p_i): integral-based cost penalising low residual energy."""
        T = 1.0
        lower = sensor.e0 - sensor.e_res
        upper = lower + tx_power * T
        cost, _ = integrate.quad(self._energy_cost_integrand, lower, upper)
        return cost / self.mu

    def calc_connectivity_benefit(self, sensor, hop):
        """f_pr(i) = beta * |V'_i|"""
        vertices = self.get_k_hop_vertices(sensor, hop)
        if len(vertices) == 0:
            return 0
        return self.beta * len(vertices)

    def calc_energy_balance_benefit(self, local_net):
        """f_e(i) = alpha * sum((E_r - E_avg)^2) / |V'|"""
        vertices = local_net['vertices']
        if len(vertices) == 0:
            return 0
        energies = [s.e_res for s in vertices]
        avg_e = sum(energies) / len(energies)
        sum_diff = sum((e - avg_e) ** 2 for e in energies)
        return self.alpha * sum_diff / len(vertices)

    def calc_utility(self, sensor, power):
        """u_i = f_pr - f_e - c_i  (assuming connectivity holds, f_i=1)."""
        return (
            self.calc_connectivity_benefit(sensor, self.hop_max)
            - self.calc_energy_balance_benefit(sensor.local_net)
            - self.calc_energy_cost(sensor, power)
        )

    # ------------------------------------------------------------------ #
    #  Topology helpers                                                   #
    # ------------------------------------------------------------------ #

    def get_k_hop_vertices(self, sensor, hop):
        """Return set of sensors reachable within *hop* hops."""
        vertices = set([sensor])
        if len(sensor.neighbors) == 0 or sensor.is_ch:
            return list(vertices)
        if hop != 1:
            for nb in sensor.neighbors:
                vertices.add(nb)
                vertices.update(self.get_k_hop_vertices(nb, hop - 1))
        else:
            for nb in sensor.neighbors:
                vertices.add(nb)
        return list(vertices)

    def get_local_graph(self, sensor, hop):
        """Build local sub-graph (vertices + adjacency matrix) for k-hop neighbourhood."""
        vertices = self.get_k_hop_vertices(sensor, hop)
        n = len(vertices)
        edges = np.zeros((n, n), dtype=int)
        for s1 in vertices:
            for s2 in vertices:
                if s1 is s2:
                    continue
                if s2 in s1.neighbors:
                    i = vertices.index(s1)
                    j = vertices.index(s2)
                    edges[i, j] = 1
        return {'vertices': vertices, 'edges': edges}

    def _dfs(self, local_net, start, visited):
        i = local_net['vertices'].index(start)
        if visited[i]:
            return
        visited[i] = True
        for j in range(len(local_net['vertices'])):
            if local_net['edges'][i, j] == 1:
                self._dfs(local_net, local_net['vertices'][j], visited)

    def check_local_connectivity(self, local_net, start_sensor):
        """Check if every node in *local_net* is reachable from *start_sensor*."""
        visited = [False] * len(local_net['vertices'])
        self._dfs(local_net, start_sensor, visited)
        return all(visited)

    def update_edges_from_local(self, local_net):
        """Write local sub-graph edges back into the global adjacency matrix."""
        verts = local_net['vertices']
        for i, si in enumerate(verts):
            for j, sj in enumerate(verts):
                if i == j:
                    continue
                self.edges[si.id, sj.id] = local_net['edges'][i, j]

    def check_potential_connectivity(self):
        """True iff every node can reach the base station at maximum power.

        At ``p_max`` every node shares the same range ``calc_comm_range(p_max)``
        — the densest topology any algorithm could ever realise. Feasibility is
        rooted at the **base station**: the layout is feasible when every node has
        a (multi-hop) path to the BS, with the BS participating as a graph node
        (the data sink). A node that cannot reach the BS even at max power is
        infeasible — no routing/clustering scheme could deliver its data.

        This is the **same connectivity definition** the coverage generator
        guarantees (``scenarios.coverage_deploy.is_connected_to_bs``), so any
        frozen scenario that passed generation also passes this gate. It is
        deliberately *not* "all nodes form one component among themselves": two
        clusters that each independently reach the BS are a valid data-collection
        layout, not an infeasible one.
        """
        n = self.num_nodes
        if n == 0:
            return True
        max_rc = self.calc_comm_range(self.p_max)
        # BFS rooted at the BS (index 0); nodes are indices 1..n. The BS is a
        # graph node, so it can relay between clusters that both reach it.
        pts = np.empty((n + 1, 2), dtype=float)
        pts[0] = (self.bs_x, self.bs_y)
        pts[1:] = [[s.x, s.y] for s in self.sensors]
        seen = np.zeros(n + 1, dtype=bool)
        seen[0] = True
        frontier = [0]
        while frontier:
            i = frontier.pop()
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            nbrs = np.nonzero((d <= max_rc) & ~seen)[0]
            seen[nbrs] = True
            frontier.extend(nbrs.tolist())
        return bool(seen[1:].all())

    # ------------------------------------------------------------------ #
    #  Edge / neighbour management                                        #
    # ------------------------------------------------------------------ #

    def connect(self, src, dst):
        """Add directed edge and register neighbour."""
        self.edges[src.id, dst.id] = 1
        src.add_neighbor(dst)

    def disconnect(self, src, dst):
        """Remove directed edge and de-register neighbour."""
        self.edges[src.id, dst.id] = 0
        src.remove_neighbor(dst)

    def discover_neighbors(self):
        """Broadcast phase: every alive node discovers neighbours within comm range."""
        for si in self.sensors:
            if not si.is_alive:
                continue
            for sj in self.sensors:
                if si is sj or not sj.is_alive:
                    continue
                if si.distance_to(sj) <= si.rc:
                    self.connect(si, sj)

    def reset_round(self):
        """Zero the adjacency matrix and reset per-round sensor state."""
        self.edges = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for s in self.sensors:
            s.reset_round()

    def create_cluster_subgraph(self):
        """Return modified edge matrix with long-range CH-CM and CH-CH links removed.

        Used for BFS layer assignment within clusters.
        """
        mod_edges = self.edges.copy()
        cm_rc = self.calc_comm_range(self.p_max / 4)
        for s in self.sensors:
            if not s.is_alive or not s.is_ch:
                continue
            for nb in s.neighbors:
                if not nb.is_ch and s.distance_to(nb) > cm_rc:
                    mod_edges[s.id, nb.id] = 0
            for ch_nb in s.ch_neighbors:
                mod_edges[s.id, ch_nb.id] = 0
        return mod_edges

    # ------------------------------------------------------------------ #
    #  Compatibility helpers for graph.py / plot.py                       #
    # ------------------------------------------------------------------ #

    def to_network_dict(self, edges=None):
        """Legacy format: {'vertices': [(x,y), ...], 'edges': np.array}."""
        return {
            'vertices': [s.pos for s in self.sensors],
            'edges': edges.copy() if edges is not None else self.edges.copy(),
        }

    def to_node_dict(self):
        """Legacy format keyed by position tuple."""
        d = {}
        for s in self.sensors:
            d[s.pos] = {
                'id': s.id,
                'neighbors': [n.pos for n in s.neighbors],
                'power': s.power,
                'rc': s.rc,
                'e_res': s.e_res,
                'Vpre': s.Vpre,
                'p0': s.p0,
                'p_ch': s.p_ch,
                'CH': s.is_ch,
                'CH_belong': s.ch_belong.pos if s.ch_belong else None,
                'CH_neighbors': [n.pos for n in s.ch_neighbors],
                'c_ch': s.c_ch,
                'c_cm': s.c_cm,
                'util': s.util,
                'local_net': s.local_net,
            }
        return d
