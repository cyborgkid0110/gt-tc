"""
EE-TCM: Energy-Efficiency Topology Control Model.

Two-stage protocol:
  1. Distributed clustering (§3.4): a node S_i qualifies as CH when
         E(S_i) > β_opt · E_toSink,   β_opt = ((r_max − r)/r_max) · E_toSink/E_0.
     Eligible nodes are subsampled to `p_ch_fraction` so CH density stays
     comparable to other benchmark protocols. Non-CH nodes join the nearest CH.
  2. Topology Control Game (§3.5): per cluster, each member runs a better-
     response power-reduction loop; a strategy is accepted only if it improves
     the EFTCG-style utility
         u_i = f_k · (α_i · (p_max − p_i)/p_max + β_i · Ē_neighbors),
         α_i = 1 − E_r(i)/E_0(i),   β_i = 1 − α_i,
     where f_k is the k-connectivity indicator on the directed graph.

Notes (see PLAN.md):
  – Uses the project's Friis energy model (NetworkModel.calc_tx_cost /
    calc_node_cost), not the paper's two-regime (E_fs / E_mp) radio model.
  – `f_k` checks **strong** connectivity on the directed cluster graph (plus
    biconnectivity on the undirected version when k=2), matching the project-
    wide connectivity convention.
  – Paper's entrance sub-game (x_i ∈ {0,1}) is implemented as a per-cluster
    mixed-strategy NE. Harvested energy f_i = 0 (nodes that stay out simply
    avoid energy cost). Entering nodes then run the power-control game.
  – Data compression (§3.4 E_saving = (1 − 1/a)·(E_P+E_T+E_R) − E_compress)
    is implemented but disabled by default (`compression_a = 1`) so the
    benchmark result isn't biased by a side channel the other algorithms lack.

Reference: Elavarasan, R., & Rajaram, A. (2024). Energy-Efficiency Topology
           Control Model for Wireless Sensor Networks in IoT. Sustainable
           Computing: Informatics and Systems, 44, 101015.
"""

import math
import random
import yaml
import networkx as nx

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot
import graph


class EETCM(BaseAlgorithm):
    """EE-TCM clustering + topology-control game algorithm."""
    family = 'clustering'

    def __init__(self, net: NetworkModel, config_path: str = 'config/ee_tcm.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self._p_ch_fraction: float = float(cfg['p_ch_fraction'])
        self.k: int = int(cfg['k_connectivity'])
        self._max_adapt_iter: int = int(cfg['max_adapt_iter'])
        self._compression_a: float = float(cfg['compression_a'])
        self._compression_overhead: float = float(cfg['compression_overhead'])

    # ------------------------------------------------------------------ #
    #  Round                                                               #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        net = self.net
        net.reset_round()

        for s in net.sensors:
            s.is_ch = False
            s.ch_neighbors = []
            s._entered = False
            if s.is_alive:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)

        alive = [s for s in net.sensors if s.is_alive]
        if not alive:
            return False

        net.discover_neighbors()

        # ---- Phase 1: clustering ------------------------------------
        self._ch_election(alive)

        num_ch = sum(1 for s in alive if s.is_ch)
        print(f'EE-TCM Round {self.t}: CHs={num_ch}, '
              f'Dead={self.dead_nodes}/{net.num_nodes}')

        self._cluster_formation()
        self._filter_neighbours()
        self._connect_unaffiliated()
        self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # ---- Phase 2: topology-control game per cluster -------------
        self._entrance_game()
        self._adapt_topology()

        # ---- Phase 3: maintenance -----------------------------------
        self._maintenance()

        if self.dead_nodes >= net.num_nodes:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Phase 1: distributed clustering (§3.4)                              #
    # ------------------------------------------------------------------ #

    def _ch_election(self, alive: list[Sensor]) -> None:
        """β_opt residual-energy eligibility + Bernoulli downsample to target
        CH density. Paper is silent on CH count — downsampling keeps benchmark
        parity with LEACH / FC-CRA / SCA-Lévy."""
        net = self.net
        r_max = max(1, self.max_rounds)
        life_frac = max(0.0, (r_max - self.t) / r_max)

        for s in alive:
            if s.e0 <= 0:
                continue
            e_to_sink = net.calc_tx_cost(net.dist_to_bs(s), 'CH')
            beta_opt = life_frac * (e_to_sink / s.e0)
            if s.e_res > beta_opt * e_to_sink:
                if random.random() < self._p_ch_fraction:
                    s.is_ch = True

        # Guarantee at least one CH per round
        if not any(s.is_ch for s in alive):
            s = max(alive, key=lambda x: x.e_res)
            s.is_ch = True

    # ------------------------------------------------------------------ #
    #  Cluster formation / repair (LEACH/FC-CRA pattern)                   #
    # ------------------------------------------------------------------ #

    def _cluster_formation(self) -> None:
        net = self.net
        for ch in net.sensors:
            if not ch.is_alive or not ch.is_ch:
                continue
            ch.power = net.p_max
            ch.rc = net.calc_comm_range(net.p_max)

            for s in net.sensors:
                if s is ch or not s.is_alive:
                    continue
                if ch.distance_to(s) > ch.rc:
                    continue

                if s.is_ch:
                    if s not in ch.ch_neighbors:
                        ch.ch_neighbors.append(s)
                        net.edges[ch.id, s.id] = 1
                else:
                    if s.ch_belong is None:
                        s.ch_belong = ch
                        net.edges[ch.id, s.id] = 1
                        ch.add_neighbor(s)
                    else:
                        if s.distance_to(s.ch_belong) > s.distance_to(ch):
                            old_ch = s.ch_belong
                            old_ch.remove_neighbor(s)
                            net.edges[old_ch.id, s.id] = 0
                            s.ch_belong = ch
                            ch.add_neighbor(s)
                            net.edges[ch.id, s.id] = 1

    def _filter_neighbours(self) -> None:
        net = self.net
        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.is_ch:
                for nb in s.neighbors[:]:
                    if nb.is_ch:
                        # CH-CH edge is semantically owned by `ch_neighbors`
                        # + matrix (set in _cluster_formation). Drop the
                        # stray `neighbors` entry created by discover_neighbors
                        # but keep the matrix entry to preserve the backbone.
                        s.remove_neighbor(nb)
                    elif nb.ch_belong is not s:
                        net.disconnect(s, nb)
            elif s.ch_belong is not None:
                for nb in s.neighbors[:]:
                    if nb.is_ch:
                        if nb is not s.ch_belong:
                            net.disconnect(s, nb)
                    elif nb.ch_belong is not s.ch_belong:
                        net.disconnect(s, nb)
            else:
                for nb in s.neighbors[:]:
                    net.disconnect(s, nb)

    def _connect_unaffiliated(self) -> None:
        net = self.net
        extended = 0
        connectivity = True
        while extended == 0 and connectivity:
            extended = 1
            for s in net.sensors:
                if not s.is_alive or s.is_ch or s.ch_belong is not None:
                    continue
                if s.power == 0:
                    continue
                extended = 0
                final_cm = None
                neighbors_temp = []

                for cm in net.sensors:
                    if cm is s or not cm.is_alive:
                        continue
                    if cm.ch_belong is None or cm.is_ch:
                        continue
                    if s.distance_to(cm) <= s.rc:
                        neighbors_temp.append(cm)
                        if s.ch_belong is None:
                            s.ch_belong = cm.ch_belong
                            final_cm = cm
                        else:
                            d_old = s.distance_to(s.ch_belong)
                            d_new = s.distance_to(cm.ch_belong)
                            if d_old > d_new:
                                s.ch_belong = cm.ch_belong
                                final_cm = cm

                if final_cm is not None:
                    net.connect(final_cm, s)
                    neighbors_temp = [nb for nb in neighbors_temp
                                      if nb.ch_belong is s.ch_belong]
                    for nb in neighbors_temp:
                        net.connect(s, nb)
                else:
                    s.power += net.p_step
                    if s.power > net.p_max:
                        s.power = 0
                        s.rc = 0
                        continue
                    net.update_comm_range(s)

    def _cleanup_cross_cluster_edges(self) -> None:
        net = self.net
        for s in net.sensors:
            for nb in s.neighbors[:]:
                if (s.ch_belong is not None and nb.ch_belong is not None
                        and s.ch_belong is not nb.ch_belong):
                    net.disconnect(s, nb)

    # ------------------------------------------------------------------ #
    #  Phase 2a: Entrance Sub-Game (§3.5)                                 #
    # ------------------------------------------------------------------ #

    def _entrance_game(self) -> None:
        """Per-cluster mixed-strategy entrance sub-game.

        Each CM decides enter (transmit) or stay out (save energy) via the
        closed-form mixed NE derived from the paper's expected utility with
        f_i = 0.  CHs always enter.
        """
        net = self.net

        clusters: dict[int, list[Sensor]] = {}
        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.is_ch:
                s._entered = True
                clusters.setdefault(s.id, []).append(s)
            elif s.ch_belong is not None:
                clusters.setdefault(s.ch_belong.id, []).append(s)

        total_entered = 0
        total_stayed_out = 0

        for ch_id, members in clusters.items():
            ch = net.sensors[ch_id]
            cms = [s for s in members if not s.is_ch]

            if not cms:
                continue

            eligible: list[tuple[Sensor, float, float]] = []
            for s in cms:
                g_j = s.e_res
                c_j = net.calc_tx_cost(s.distance_to(ch), 'CM')
                if g_j > c_j:
                    eligible.append((s, g_j, c_j))

            n = len(eligible)
            if n == 0:
                total_stayed_out += len(cms)
                continue

            if n == 1:
                eligible[0][0]._entered = True
                total_entered += 1
                total_stayed_out += len(cms) - 1
                continue

            log_prod = sum(math.log(c_j / g_j) for _, g_j, c_j in eligible)
            R = math.exp(log_prod / (n - 1))

            for s, g_j, c_j in eligible:
                p_j = 1.0 - R * g_j / c_j
                p_j = max(0.0, min(1.0, p_j))

                if random.random() < p_j:
                    s._entered = True
                    total_entered += 1
                else:
                    total_stayed_out += 1

            total_stayed_out += len(cms) - n

        print(f'  entrance game: {total_entered} entered, '
              f'{total_stayed_out} stayed out')

    # ------------------------------------------------------------------ #
    #  Phase 2b: Topology Control Game (§3.5)                             #
    # ------------------------------------------------------------------ #

    def _neighbor_energy(self, sensor: Sensor,
                         active_ids: set[int] | None = None) -> float:
        """Ē_i(p_i) = avg [E_r(j)/E_0(j)] over alive 1-hop neighbours."""
        nbs = [nb for nb in sensor.neighbors if nb.is_alive and nb.e0 > 0]
        if active_ids is not None:
            nbs = [nb for nb in nbs if nb.id in active_ids]
        if not nbs:
            return 0.0
        return sum(nb.e_res / nb.e0 for nb in nbs) / len(nbs)

    def _utility(self, sensor: Sensor, f_k: float,
                 active_ids: set[int] | None = None) -> float:
        """u_i = f_k · (α_i · (p_max − p_i)/p_max + β_i · Ē_i)."""
        net = self.net
        alpha_i = 1.0 - (sensor.e_res / sensor.e0) if sensor.e0 > 0 else 1.0
        beta_i = 1.0 - alpha_i
        power_saving = (net.p_max - sensor.power) / net.p_max
        return f_k * (alpha_i * power_saving
                      + beta_i * self._neighbor_energy(sensor, active_ids))

    def _cluster_digraph(self, members: list[Sensor]) -> nx.DiGraph:
        """Directed cluster subgraph over member ids, including the CH."""
        net = self.net
        ids = {s.id for s in members}
        D = nx.DiGraph()
        for s in members:
            D.add_node(s.id)
        for s in members:
            for nb in s.neighbors:
                if nb.id in ids:
                    D.add_edge(s.id, nb.id)
        return D

    def _f_k(self, D: nx.DiGraph) -> float:
        """k-connectivity indicator on the directed cluster graph."""
        if D.number_of_nodes() < 2:
            return 0.0
        if not nx.is_strongly_connected(D):
            return 0.0
        if self.k >= 2 and not nx.is_biconnected(D.to_undirected()):
            return 0.0
        return 1.0

    def _adapt_topology(self) -> None:
        """Sequential better-response power reduction, scoped per cluster.

        Mirrors EFTCG._adapt's fast/slow-path split: if a candidate reduction
        loses no outgoing link the utility's connectivity term is preserved
        and the power-saving term strictly improves for α_i > 0 → auto-accept.
        Otherwise, evaluate f_k before/after on the cluster subgraph and
        accept only when utility improves.
        """
        net = self.net

        # group members per CH
        clusters: dict[int, list[Sensor]] = {}
        for s in net.sensors:
            if not s.is_alive:
                continue
            ch = s if s.is_ch else s.ch_belong
            if ch is None:
                continue
            clusters.setdefault(ch.id, []).append(s)

        # Per-cluster iteration budget: each cluster runs its own game G_i
        # (paper §3.5), so they must not share a single global counter. A
        # dense cluster processed first would otherwise starve later ones.
        total_steps = 0
        for ch_id, members in clusters.items():
            active = [s for s in members if s.is_ch or s._entered]
            if len(active) < 2:
                continue
            active_ids = {s.id for s in active}
            cluster_steps = 0
            converged = False
            while not converged and cluster_steps < self._max_adapt_iter:
                converged = True
                for sensor in active:
                    if sensor.is_ch:
                        continue  # CH stays at p_max for intra-cluster reach
                    if not sensor.is_alive or sensor.power <= net.p_min:
                        continue

                    new_power = max(round(sensor.power - net.p_step, 12),
                                     net.p_min)
                    if new_power >= sensor.power:
                        continue
                    new_rc = net.calc_comm_range(new_power)

                    links_lost = [nb for nb in sensor.neighbors
                                  if nb.id in active_ids
                                  and sensor.distance_to(nb) > new_rc]

                    if not links_lost:
                        alpha_i = (1.0 - sensor.e_res / sensor.e0
                                   if sensor.e0 > 0 else 1.0)
                        if alpha_i > 0:
                            sensor.power = new_power
                            sensor.rc = new_rc
                            converged = False
                        cluster_steps += 1
                        total_steps += 1
                        continue

                    D_cur = self._cluster_digraph(active)
                    old_util = self._utility(sensor, self._f_k(D_cur),
                                             active_ids)

                    for nb in links_lost:
                        sensor.remove_neighbor(nb)
                        net.edges[sensor.id, nb.id] = 0
                    old_power, old_rc = sensor.power, sensor.rc
                    sensor.power, sensor.rc = new_power, new_rc

                    D_trial = self._cluster_digraph(active)
                    new_util = self._utility(sensor, self._f_k(D_trial),
                                             active_ids)

                    if new_util > old_util:
                        converged = False
                    else:
                        sensor.power, sensor.rc = old_power, old_rc
                        for nb in links_lost:
                            sensor.add_neighbor(nb)
                            net.edges[sensor.id, nb.id] = 1

                    cluster_steps += 1
                    total_steps += 1

        print(f'  adapt: {total_steps} power-update passes')

    # ------------------------------------------------------------------ #
    #  Phase 3: maintenance                                                #
    # ------------------------------------------------------------------ #

    def _maintenance(self) -> None:
        """Charge per-round maintenance energy along the cluster routing tree
        (CM->relay->CH->backbone->BS) via the shared cluster-tree helpers.
        CM costs are optionally reduced by the compression factor 1/a."""
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
