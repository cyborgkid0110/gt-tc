"""
EFTCG: Energy-Efficient and Fault-Tolerant Topology Control Game.

Non-cooperative power-control game where each node adjusts transmit power to
balance energy efficiency against k-connectivity (single or biconnected).

Utility function:
  u_i(p_i, p_{-i}) = f_k(p_i, p_{-i}) * [α_i * (p_max - p_i)/p_max
                                          + β_i * E_i(p_i)]

where:
  - f_k ∈ {0, 1}: k-connectivity indicator (1 if network is k-connected)
  - α_i = 1 - E_r(i)/E_0(i): self-adaptive weight (depletes → prioritise saving)
  - β_i = 1 - α_i
  - E_i(p_i) = avg [E_r(j)/E_0(j)] over one-hop neighbours at power p_i

Two variants differ only in k:
  - EFTCG-1: k=1 (single connectivity), energy-efficient
  - EFTCG-2: k=2 (biconnected, no cut-points), fault-tolerant at slight cost

Note on connectivity: the paper assumes bidirectional links. This project treats
unidirectional links as valid connectivity. k-connectivity is checked on the
undirected version of the graph (union of forward and reverse edges).

Note on energy model: the paper does not model per-round energy dissipation.
This implementation adds a maintenance phase using the shared NetworkModel
energy model (role='CM', clustering=False), same as DIA-MIA and TCLE.

Reference: Li et al., "EFTCG: An Energy Efficient and Fault Tolerant Topology
Control Game for Wireless Sensor Networks", Sensors, 2017.
"""

import yaml
import numpy as np
import networkx as nx

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot, tx_power_plot


class EFTCG(BaseAlgorithm):
    """EFTCG Topology Control Game algorithm (EFTCG-1 or EFTCG-2)."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/eftcg.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # k=1: single connected, k=2: biconnected
        self.k: int = cfg['k_connectivity']

        # Flag: adapt topology before the next maintenance round
        self._needs_adapt: bool = True

        # Safety limit for adaptation convergence loop
        self._max_adapt_iter: int = 300_000

    # ------------------------------------------------------------------ #
    #  Single round                                                        #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        """Execute one simulation round. Returns False when network is dead."""
        net = self.net

        # Re-adapt if topology changed (node died or first round)
        if self._needs_adapt:
            self._initialize_topology()
            self._adapt()
            self._needs_adapt = False

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())
            tx_power_plot(net.to_network_dict(), net.to_node_dict())

        # Maintenance: energy deduction
        self._maintenance()

        print(f'EFTCG-{self.k} Round {self.t}: '
              f'Dead={self.dead_nodes}/{net.num_nodes}')

        if self.dead_nodes >= net.num_nodes:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Topology initialisation                                             #
    # ------------------------------------------------------------------ #

    def _initialize_topology(self) -> None:
        """Set all alive nodes to p_max and build the maximum-power graph g_max."""
        net = self.net

        # Zero edge matrix and clear all neighbour lists
        net.edges = np.zeros((net.num_nodes, net.num_nodes), dtype=int)
        for s in net.sensors:
            s.neighbors = []

        # Set every alive node to p_max
        for s in net.sensors:
            if s.is_alive:
                s.power = net.p_max
                s.rc = net.calc_comm_range(net.p_max)

        # Discover all neighbours at max power (builds g_max)
        net.discover_neighbors()

        alive_count = sum(1 for s in net.sensors if s.is_alive)
        print(f'EFTCG-{self.k}: initialised g_max ({alive_count} alive nodes)')

        # Warn if g_max itself is not k-connected (EFTCG-2 may be infeasible)
        G = self._build_directed_graph()
        if not self._check_k_connectivity(G):
            print(f'EFTCG-{self.k}: WARNING — g_max is not {self.k}-connected; '
                  f'target connectivity may not be achievable')

    # ------------------------------------------------------------------ #
    #  k-connectivity                                                      #
    # ------------------------------------------------------------------ #

    def _build_directed_graph(self) -> nx.DiGraph:
        """Build directed graph from net.edges for strong connectivity check."""
        net = self.net
        D = nx.DiGraph()

        for s in net.sensors:
            if s.is_alive:
                D.add_node(s.id)

        for i in range(net.num_nodes):
            if not net.sensors[i].is_alive:
                continue
            for j in range(net.num_nodes):
                if i == j or not net.sensors[j].is_alive:
                    continue
                if net.edges[i, j] == 1:
                    D.add_edge(i, j)

        return D

    def _check_k_connectivity(self, D: nx.DiGraph) -> bool:
        """Return True if the directed graph satisfies k-connectivity.

        Connectivity is checked on the directed graph (strong connectivity)
        to ensure every node can both send to and receive from every other
        via multi-hop directed paths. For k=2 biconnectivity, the undirected
        version is additionally checked for no cut-points.
        """
        if D.number_of_nodes() < 2:
            return False
        if not nx.is_strongly_connected(D):
            return False
        if self.k >= 2:
            # Biconnectivity (no cut-points) on undirected version
            G = D.to_undirected()
            return nx.is_biconnected(G)
        return True

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _compute_neighbor_energy(self, sensor: Sensor) -> float:
        """E_i(p_i) = avg [E_r(j)/E_0(j)] over one-hop neighbours at p_i."""
        neighbors = [nb for nb in sensor.neighbors if nb.is_alive]
        if not neighbors:
            return 0.0
        return sum(nb.e_res / nb.e0 for nb in neighbors if nb.e0 > 0) / len(neighbors)

    def _utility(self, sensor: Sensor, f_k: float) -> float:
        """u_i = f_k * [α_i * (p_max - p_i)/p_max + β_i * E_i(p_i)]."""
        net = self.net
        alpha_i = 1.0 - (sensor.e_res / sensor.e0) if sensor.e0 > 0 else 1.0
        beta_i = 1.0 - alpha_i
        power_saving = (net.p_max - sensor.power) / net.p_max
        neighbor_energy = self._compute_neighbor_energy(sensor)
        return f_k * (alpha_i * power_saving + beta_i * neighbor_energy)

    # ------------------------------------------------------------------ #
    #  Adaptation: sequential better-response (Phase 2 from paper)        #
    # ------------------------------------------------------------------ #

    def _adapt(self) -> None:
        """Sequential better-response ordered by node ID.

        Each node decrements power by one step (p_step) and accepts only if
        utility strictly improves. Iterate passes until no node updates (NE).

        Fast path: if no outgoing link is lost AND α_i > 0, utility strictly
        improves — auto-accept without graph rebuild.
        (At full energy α_i = 0, so no improvement → first adaptation stays
        at p_max, which is correct NE behaviour before any energy depletion.)
        """
        net = self.net
        total_steps = 0

        converged = False
        while not converged and total_steps < self._max_adapt_iter:
            converged = True

            for sensor in net.sensors:
                if not sensor.is_alive:
                    continue
                if sensor.power <= net.p_min:
                    continue

                new_power = max(round(sensor.power - net.p_step, 6), net.p_min)
                if new_power >= sensor.power:
                    continue

                new_rc = net.calc_comm_range(new_power)

                # Identify outgoing links that would be lost
                links_lost = [nb for nb in sensor.neighbors
                              if sensor.distance_to(nb) > new_rc]

                if not links_lost:
                    # FAST PATH: topology unchanged, check if utility improves.
                    # Δu = f_k * α_i * (old_power - new_power) / p_max
                    # Strictly positive iff α_i > 0 (i.e. e_res < e0).
                    alpha_i = (1.0 - sensor.e_res / sensor.e0
                               if sensor.e0 > 0 else 1.0)
                    if alpha_i > 0:
                        sensor.power = new_power
                        sensor.rc = new_rc
                        converged = False
                    total_steps += 1
                    continue

                # SLOW PATH: a link would be lost; evaluate utility change.
                # Compute current utility.
                G_current = self._build_directed_graph()
                f_k_current = 1.0 if self._check_k_connectivity(G_current) else 0.0
                old_util = self._utility(sensor, f_k_current)

                # Temporarily remove lost outgoing links and apply new power.
                for nb in links_lost:
                    sensor.remove_neighbor(nb)
                    net.edges[sensor.id, nb.id] = 0

                old_power = sensor.power
                old_rc = sensor.rc
                sensor.power = new_power
                sensor.rc = new_rc

                # Compute trial utility.
                G_trial = self._build_directed_graph()
                f_k_trial = 1.0 if self._check_k_connectivity(G_trial) else 0.0
                new_util = self._utility(sensor, f_k_trial)

                if new_util > old_util:
                    # Accept: keep reduced power and pruned links
                    converged = False
                else:
                    # Reject: restore links and power
                    sensor.power = old_power
                    sensor.rc = old_rc
                    for nb in links_lost:
                        sensor.add_neighbor(nb)
                        net.edges[sensor.id, nb.id] = 1

                total_steps += 1

        print(f'EFTCG-{self.k} converged in {total_steps} steps '
              f'(converged={converged})')

    # ------------------------------------------------------------------ #
    #  Maintenance: energy deduction                                       #
    # ------------------------------------------------------------------ #

    def _maintenance(self) -> None:
        """Deduct one round of energy from all alive nodes.

        Each EFTCG node acts as a sensing + transmitting node (CM role,
        no aggregation). Distance = sensor.rc (post-adaptation comm range).
        """
        net = self.net

        for s in net.sensors:
            if not s.is_alive:
                continue

            cost = net.calc_node_cost(s, 'CM', clustering=False)
            s.c_cm = cost
            s.e_res -= cost

            if s.e_res <= 0:
                self._track_death(s)
                # Topology changed: re-adapt before the next maintenance round
                self._needs_adapt = True
