"""
GT2: Game-Theoretic Topology Control algorithm.

Two-stage non-cooperative game:
  Game 1 — Mixed-strategy clustering game for CH election and cluster formation.
  Game 2 — Pure-strategy power-control game for intra-cluster topology optimisation.
"""

import math
import random
import yaml

from algos import BaseAlgorithm
from model import NetworkModel
from plot import directional_wsn_plot, cluster_head_probability_plot, tx_power_plot
import graph


class GT2(BaseAlgorithm):
    """Game-Theoretic Topology Control (two-stage) algorithm."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/gt2.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        # load GT2-specific parameters
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.payoff = cfg['payoff']
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']
        self.mu = cfg['mu']

        # push game-2 weights into the network model
        self.net.alpha = self.alpha
        self.net.beta = self.beta
        self.net.mu = self.mu

    # ------------------------------------------------------------------ #
    #  Single round                                                        #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        """Run one simulation round. Returns False when the network is dead."""
        net = self.net

        # ---- reset --------------------------------------------------
        net.reset_round()

        # ---- Phase 1: neighbour discovery ----------------------------
        net.discover_neighbors()

        # ---- Phase 2: clustering game --------------------------------
        ch_can, ch_true = self._clustering_game()
        if ch_true == 0:
            self._maintenance_no_cluster()
            if self.dead_nodes >= net.num_nodes:
                return False
            return True

        # ---- Phase 3: cluster formation ------------------------------
        self._cluster_formation()

        # ---- Phase 4: filter cross-cluster edges ---------------------
        self._filter_neighbours()

        # ---- Phase 5: connect unaffiliated nodes ---------------------
        self._connect_unaffiliated()
        self._cleanup_cross_cluster_edges()

        # ---- build layer structure (pre-adaptation) ------------------
        mod_edges = net.create_cluster_subgraph()
        mod_net_dict = net.to_network_dict(edges=mod_edges)
        node_dict = net.to_node_dict()

        if self.t % self.plot_period == 0:
            cluster_head_probability_plot(mod_net_dict, node_dict)
            directional_wsn_plot(net.to_network_dict(), node_dict)

        G = graph.build_graph(mod_net_dict['vertices'], mod_net_dict['edges'])
        layered_batches = graph.divide_network_by_clusters(G, node_dict)

        # ---- Phase 6: power-control game -----------------------------
        self._power_control_game(layered_batches)

        print(f'Iteration {self.t}: Finished, '
              f'Candidate CH: {ch_can}, Real CH: {ch_true}')

        # ---- rebuild layer structure (pre-maintenance) ---------------
        mod_edges = net.create_cluster_subgraph()
        mod_net_dict = net.to_network_dict(edges=mod_edges)
        node_dict = net.to_node_dict()

        if self.t % self.plot_period == 0:
            tx_power_plot(mod_net_dict, node_dict)
            directional_wsn_plot(net.to_network_dict(), node_dict)

        G = graph.build_graph(mod_net_dict['vertices'], mod_net_dict['edges'])
        layered_batches = graph.divide_network_by_clusters(G, node_dict)

        # ---- Phase 7: maintenance (energy deduction) -----------------
        self._maintenance(layered_batches)

        if self.dead_nodes >= net.num_nodes:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Phase 2: Clustering game                                            #
    # ------------------------------------------------------------------ #

    def _clustering_game(self) -> tuple[int, int]:
        """Mixed-strategy CH election.

        Returns (candidates_count, elected_ch_count).
        """
        net = self.net
        ch_can = 0
        ch_true = 0

        for s in net.sensors:
            if not s.is_alive or len(s.neighbors) == 0:
                continue

            # previous-round CHs: reset and skip
            if s.is_ch:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)
                s.ch_neighbors = []
                s.is_ch = False
                continue

            c_ch = net.calc_node_cost(s, 'CH', clustering=True)
            c_cm = net.calc_node_cost(s, 'CM', clustering=True)
            s.c_ch = c_ch
            s.c_cm = c_cm

            if c_ch - c_cm < 0:
                p0 = 1
            else:
                p0 = 1 - pow((c_ch - c_cm) / (self.payoff - c_cm),
                             1 / len(s.neighbors))
            s.p0 = p0

            if random.random() < p0:
                s.p_ch = p0
                if random.uniform(0, 1) < p0:
                    s.is_ch = True
                    ch_true += 1
                ch_can += 1

        return ch_can, ch_true

    # ------------------------------------------------------------------ #
    #  Phase 3: Cluster formation                                          #
    # ------------------------------------------------------------------ #

    def _cluster_formation(self):
        """CHs advertise; CMs join the nearest CH."""
        net = self.net

        for ch in net.sensors:
            if not ch.is_alive or not ch.is_ch:
                continue

            ch.power = net.p_max
            ch.rc = net.calc_comm_range(net.p_max)
            ch.c_ch = net.calc_node_cost(ch, 'CH', clustering=False)

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

    # ------------------------------------------------------------------ #
    #  Phase 4: Neighbour filtering                                        #
    # ------------------------------------------------------------------ #

    def _filter_neighbours(self):
        """Remove edges between nodes in different clusters."""
        net = self.net

        for s in net.sensors:
            if not s.is_alive:
                continue

            if s.is_ch:
                for nb in s.neighbors[:]:
                    if nb.is_ch:
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

    # ------------------------------------------------------------------ #
    #  Phase 5: Connect unaffiliated nodes                                 #
    # ------------------------------------------------------------------ #

    def _connect_unaffiliated(self):
        """Unaffiliated nodes incrementally raise power to reach a cluster."""
        net = self.net
        extended = 0
        connectivity = True

        while extended == 0 and connectivity:
            extended = 1

            for s in net.sensors:
                if not s.is_alive or s.is_ch or s.ch_belong is not None:
                    continue

                extended = 0    # still unjoined nodes exist

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
                        s.power = net.p_max
                        print("Reach p_max", s.id)
                        connectivity = False
                    net.update_comm_range(s)

    def _cleanup_cross_cluster_edges(self):
        net = self.net
        for s in net.sensors:
            for nb in s.neighbors[:]:
                if (s.ch_belong is not None and nb.ch_belong is not None
                        and s.ch_belong is not nb.ch_belong):
                    net.disconnect(s, nb)

    # ------------------------------------------------------------------ #
    #  Phase 6: Power-control game                                         #
    # ------------------------------------------------------------------ #

    def _power_control_game(self, layered_batches: dict):
        """Iterative best-response power reduction toward Nash equilibrium."""
        net = self.net

        for _ch_pos, layers in layered_batches.items():
            nash_eq = False
            while not nash_eq:
                nash_eq = True
                for i, layer in reversed(list(enumerate(layers))):
                    if i == 0:
                        continue
                    for batch in layer:
                        for node_pos in batch:
                            s = net.sensor_by_pos(
                                tuple(float(x) for x in node_pos))
                            if s.is_ch:
                                continue

                            if s.local_net is None:
                                s.local_net = net.get_local_graph(
                                    s, net.hop_max)
                            if s.util is None:
                                s.util = net.calc_utility(s, s.power)

                            new_power = max(s.power - net.p_step, net.p_min)
                            new_rc = net.calc_comm_range(new_power)

                            old_neighbors = s.neighbors[:]
                            topology_changed = False

                            for nb in s.neighbors[:]:
                                if s.distance_to(nb) > new_rc:
                                    topology_changed = True
                                    s.remove_neighbor(nb)

                            new_local = net.get_local_graph(s, net.hop_max)

                            if not net.check_local_connectivity(new_local, s):
                                new_util = (-1e6
                                            * net.calc_energy_cost(s,
                                                                   new_power))
                            else:
                                new_util = net.calc_utility(s, new_power)

                            if new_util > s.util:
                                s.util = new_util
                                s.power = new_power
                                s.rc = new_rc
                                if topology_changed:
                                    s.local_net = {
                                        'vertices': new_local['vertices'],
                                        'edges': new_local['edges'].copy(),
                                    }
                                    net.update_edges_from_local(new_local)
                                nash_eq = False
                            else:
                                s.neighbors = old_neighbors

    # ------------------------------------------------------------------ #
    #  Phase 7: Maintenance                                                #
    # ------------------------------------------------------------------ #

    def _maintenance(self, layered_batches: dict):
        """Deduct energy using routing-based per-hop TX cost."""
        net = self.net

        routing_tree = net.build_routing_tree()
        costs = net.compute_maintenance_costs(routing_tree)

        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.is_ch:
                info = routing_tree.get(s.id)
                tx_dist = info['tx_dist'] if info else math.hypot(s.x, s.y)
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

    def _maintenance_no_cluster(self):
        """Deduct energy using routing-based per-hop TX cost (no CHs)."""
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
                    self._track_death(s)
