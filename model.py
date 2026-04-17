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
        self.edges = np.zeros((self.num_nodes, self.num_nodes), dtype=int)

        self._pos_map = {s.pos: s for s in sensors}

        # Radio parameters
        self.pth = params.get('pth', 7e-10)
        self.wave = params.get('wave', 0.1224)
        self.p_min = params.get('p_min', 0.01)
        self.p_max = params.get('p_max', 0.08)
        self.p_step = params.get('p_step', 0.0001)
        self.hop_max = params.get('hop_max', 3)

        # Energy parameters
        self.e_elec = params.get('e_elec', 50e-9)
        self.e_agg = params.get('e_agg', 5e-9)
        self.m_pkt_s = params.get('m_pkt_s', 20)
        self.m_pkt_l = params.get('m_pkt_l', 1000)

        # Game 2 parameters
        self.alpha = params.get('alpha', 1.5)
        self.beta = params.get('beta', 0.1)
        self.mu = params.get('mu', 0.01)

        for s in self.sensors:
            s.rc = self.calc_comm_range(s.power)

    def sensor_by_pos(self, pos):
        """Look up a sensor by its position tuple."""
        return self._pos_map.get(pos)

    # ------------------------------------------------------------------ #
    #  Communication range (Friis free-space model)                       #
    # ------------------------------------------------------------------ #

    def calc_comm_range(self, power):
        """R_tx from Friis equation."""
        return math.sqrt((power * self.wave ** 2) / (self.pth * 16 * math.pi ** 2))

    def calc_rx_power(self, p_tx, d):
        """Received power at distance d."""
        return (p_tx * self.wave ** 2) / (16 * math.pi ** 2 * d ** 2)

    def update_comm_range(self, sensor):
        sensor.rc = self.calc_comm_range(sensor.power)

    # ------------------------------------------------------------------ #
    #  Energy model                                                       #
    # ------------------------------------------------------------------ #

    def calc_tx_cost(self, d, role, layer_depth=1):
        """Transmission energy with Friis-based amplifier model."""
        m_bit = self.m_pkt_s if role == 'CM' else self.m_pkt_l
        t_tx = 1e-6
        return m_bit * layer_depth * (
            self.e_elec + (4 * math.pi / self.wave) ** 2 * self.pth * t_tx * d ** 2
        )

    def calc_node_cost(self, sensor, role, clustering, layer_depth=1):
        """Total energy cost for a sensor in a given role.

        When *clustering* is True the distance is measured to the sink
        (origin).  Otherwise the sensor's current communication range
        is used as the reference distance.
        """
        if clustering:
            d = math.hypot(sensor.x, sensor.y)
        else:
            d = sensor.rc

        c_tx = self.calc_tx_cost(d, role, layer_depth)

        if role == 'CM':
            m_bit = 8
            i_sense = random.uniform(1e-8, 5e-7)
            c_sense = sensor.Vpre * i_sense * m_bit
            c_process = sensor.Vpre * m_bit * i_sense / 4
            return c_sense + c_process + c_tx
        else:   # CH
            c_rx = self.m_pkt_l * self.e_elec
            c_agg = self.m_pkt_l * self.e_agg
            return c_rx + c_agg + c_tx

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
        """Can every node pair potentially connect at maximum power?"""
        max_rc = self.calc_comm_range(self.p_max)
        connected = set()
        for i, si in enumerate(self.sensors):
            for j, sj in enumerate(self.sensors):
                if i >= j:
                    continue
                if si.distance_to(sj) <= max_rc:
                    connected.add(i)
                    connected.add(j)
        return len(connected) == self.num_nodes

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
