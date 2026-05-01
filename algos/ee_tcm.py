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
  – Paper's entrance/harvesting sub-game (x_i ∈ {0,1} with harvested energy
    f_i) is not materialised: the topology-control game replaces the discrete
    decision with a continuous p_i ∈ [p_min, p_max] — EFTCG's canonical form.
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

    def __init__(self, net: NetworkModel, config_path: str = 'config/ee_tcm.yaml'):
        super().__init__(net, config_path)

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

        print(f'EE-TCM Round {self.t} start. '
              f'Dead: {self.dead_nodes}/{net.num_nodes}')

        for s in net.sensors:
            s.is_ch = False
            s.ch_neighbors = []
            if s.is_alive:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)

        alive = [s for s in net.sensors if s.is_alive]
        if not alive:
            return False

        net.discover_neighbors()

        # ---- Phase 1: clustering ------------------------------------
        self._ch_election(alive)
        self._cluster_formation()
        self._filter_neighbours()
        self._connect_unaffiliated()
        self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # ---- Phase 2: topology-control game per cluster -------------
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
            e_to_sink = net.calc_tx_cost(math.hypot(s.x, s.y), 'CH')
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
    #  Phase 2: Topology Control Game (§3.5)                              #
    # ------------------------------------------------------------------ #

    def _neighbor_energy(self, sensor: Sensor) -> float:
        """Ē_i(p_i) = avg [E_r(j)/E_0(j)] over alive 1-hop neighbours."""
        nbs = [nb for nb in sensor.neighbors if nb.is_alive and nb.e0 > 0]
        if not nbs:
            return 0.0
        return sum(nb.e_res / nb.e0 for nb in nbs) / len(nbs)

    def _utility(self, sensor: Sensor, f_k: float) -> float:
        """u_i = f_k · (α_i · (p_max − p_i)/p_max + β_i · Ē_i)."""
        net = self.net
        alpha_i = 1.0 - (sensor.e_res / sensor.e0) if sensor.e0 > 0 else 1.0
        beta_i = 1.0 - alpha_i
        power_saving = (net.p_max - sensor.power) / net.p_max
        return f_k * (alpha_i * power_saving + beta_i * self._neighbor_energy(sensor))

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
            if len(members) < 2:
                continue
            cluster_steps = 0
            converged = False
            while not converged and cluster_steps < self._max_adapt_iter:
                converged = True
                for sensor in members:
                    if sensor.is_ch:
                        continue  # CH stays at p_max for intra-cluster reach
                    if not sensor.is_alive or sensor.power <= net.p_min:
                        continue

                    new_power = max(round(sensor.power - net.p_step, 6),
                                     net.p_min)
                    if new_power >= sensor.power:
                        continue
                    new_rc = net.calc_comm_range(new_power)

                    links_lost = [nb for nb in sensor.neighbors
                                  if sensor.distance_to(nb) > new_rc]

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

                    D_cur = self._cluster_digraph(members)
                    old_util = self._utility(sensor, self._f_k(D_cur))

                    for nb in links_lost:
                        sensor.remove_neighbor(nb)
                        net.edges[sensor.id, nb.id] = 0
                    old_power, old_rc = sensor.power, sensor.rc
                    sensor.power, sensor.rc = new_power, new_rc

                    D_trial = self._cluster_digraph(members)
                    new_util = self._utility(sensor, self._f_k(D_trial))

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
        """Deduct energy for one round of data flow, using the shared layered-
        batch pattern. CH charge bypasses `calc_node_cost('CH')`'s tx leg to
        avoid double-charging with the explicit CH→BS hop (same pattern as
        FC-CRA). CM maintenance is optionally reduced by the compression
        factor 1/a and offset by the compression overhead."""
        net = self.net

        mod_edges = net.create_cluster_subgraph()
        mod_net_dict = net.to_network_dict(edges=mod_edges)
        node_dict = net.to_node_dict()
        G = graph.build_graph(mod_net_dict['vertices'], mod_net_dict['edges'])
        layered_batches = graph.divide_network_by_clusters(G, node_dict)

        # Compression gate: `compression_a == 1` means disabled; in that mode
        # the overhead is also skipped so a nonzero `compression_overhead` in
        # the config doesn't silently perturb the energy accounting.
        compression_enabled = self._compression_a > 1
        cm_compression = (1.0 / self._compression_a
                          if compression_enabled else 1.0)
        cm_overhead = self._compression_overhead if compression_enabled else 0.0

        for ch_pos, layers in layered_batches.items():
            ch = net.sensor_by_pos(tuple(float(x) for x in ch_pos))
            if ch is None or not ch.is_alive:
                continue

            depth = len(layers)
            # CH: reception + aggregation only (tx charged per-hop below).
            ch.c_ch = net.m_pkt_l * (net.e_elec + net.e_agg)
            ch.e_res -= ch.c_ch
            if ch.e_res <= 0:
                self._track_death(ch)

            for i, layer in enumerate(layers):
                if i == 0:
                    continue
                for batch in layer:
                    for node_pos in batch:
                        s = net.sensor_by_pos(
                            tuple(float(x) for x in node_pos))
                        if s is None or not s.is_alive:
                            continue
                        cost = net.calc_node_cost(
                            s, 'CM', clustering=False,
                            layer_depth=depth - i)
                        cost = cost * cm_compression + cm_overhead
                        s.c_cm = cost
                        s.e_res -= cost
                        if s.e_res <= 0:
                            self._track_death(s)

        # CH → BS (direct single hop; inter-cluster routing is out of scope
        # for EE-TCM per the paper — the topology-control game optimises only
        # intra-cluster transmit powers).
        for ch_pos in layered_batches:
            ch = net.sensor_by_pos(tuple(float(x) for x in ch_pos))
            if ch is None or not ch.is_alive:
                continue
            d = math.hypot(ch.x, ch.y)
            tx = net.calc_tx_cost(d, 'CH')
            ch.e_res -= tx
            if ch.e_res <= 0:
                self._track_death(ch)
