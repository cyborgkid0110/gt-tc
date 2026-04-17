"""
LEACH: Low-Energy Adaptive Clustering Hierarchy.

Randomized cluster-head rotation with threshold-based election,
bounded-range cluster formation, and single-round data transmission.

Reference: Heinzelman, Chandrakasan & Balakrishnan (2000).
"""

import random
import yaml

from algos import BaseAlgorithm
from model import NetworkModel
from plot import directional_wsn_plot


class LEACH(BaseAlgorithm):
    """LEACH clustering protocol."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/leach.yaml'):
        super().__init__(net, config_path)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.p_ch_fraction = cfg['p_ch_fraction']

        # CH rotation: track which nodes have been CH in current 1/P cycle
        self._ch_history: set[int] = set()
        self._cycle_length = int(1 / self.p_ch_fraction)

    # ------------------------------------------------------------------ #
    #  Round                                                               #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        net = self.net
        net.reset_round()

        print(f'LEACH Round {self.t} start. Dead nodes: {self.dead_nodes}/{net.num_nodes}')

        # LEACH manages rotation via _ch_history, so clear is_ch each round
        # and reset non-CH power to p_max/4
        for s in net.sensors:
            s.is_ch = False
            s.ch_neighbors = []
            if s.is_alive:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)

        # Phase 1: CH election
        num_ch = self._ch_election()

        if num_ch != 0:
            # Phase 2: neighbour discovery + cluster formation
            net.discover_neighbors()
            self._cluster_formation()
            self._filter_neighbours()
            self._connect_unaffiliated()
            self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # Phase 3: steady-state (energy deduction)
        self._steady_state()

        # update rotation history
        self._update_ch_history()

        if self.dead_nodes >= net.num_nodes:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Phase 1: CH election                                                #
    # ------------------------------------------------------------------ #

    def _ch_election(self) -> int:
        """LEACH threshold T(n) election. Returns number of elected CHs."""
        net = self.net
        P = self.p_ch_fraction
        cycle_len = self._cycle_length
        num_ch = 0

        for s in net.sensors:
            if not s.is_alive:
                continue

            # G: nodes not yet CH in this 1/P cycle
            if s.id in self._ch_history:
                continue

            r_mod = self.t % cycle_len
            denom = 1 - P * r_mod
            threshold = P / denom if denom > 0 else 1.0

            if random.random() < threshold:
                s.is_ch = True
                num_ch += 1

        return num_ch

    # ------------------------------------------------------------------ #
    #  Phase 2: cluster formation (bounded range, same as GT2)             #
    # ------------------------------------------------------------------ #

    def _cluster_formation(self):
        """CHs advertise at max power; CMs within range join nearest CH."""
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

    def _connect_unaffiliated(self):
        """Unaffiliated nodes raise power to reach a cluster.

        If a node reaches p_max without connecting, its power is set to 0
        and it is skipped.
        """
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
                        s.power = 0
                        s.rc = 0
                        continue
                    net.update_comm_range(s)

    def _cleanup_cross_cluster_edges(self):
        net = self.net
        for s in net.sensors:
            for nb in s.neighbors[:]:
                if (s.ch_belong is not None and nb.ch_belong is not None
                        and s.ch_belong is not nb.ch_belong):
                    net.disconnect(s, nb)

    # ------------------------------------------------------------------ #
    #  Phase 3: steady-state (energy deduction)                            #
    # ------------------------------------------------------------------ #

    def _steady_state(self):
        """Deduct energy for one round of data transmission."""
        net = self.net

        for s in net.sensors:
            if not s.is_alive:
                continue

            if s.is_ch:
                s.c_ch = net.calc_node_cost(s, 'CH', clustering=False)
                s.e_res -= s.c_ch

                if s.e_res <= 0:
                    self._track_death(s)

            elif s.ch_belong is not None:
                s.c_cm = net.calc_node_cost(s, 'CM', clustering=False)
                s.e_res -= s.c_cm

                if s.e_res <= 0:
                    self._track_death(s)

    # ------------------------------------------------------------------ #
    #  CH rotation history                                                 #
    # ------------------------------------------------------------------ #

    def _update_ch_history(self):
        """Record this round's CHs; clear at cycle boundary."""
        for s in self.net.sensors:
            if s.is_ch:
                self._ch_history.add(s.id)

        # end of 1/P cycle → all nodes eligible again
        if (self.t + 1) % self._cycle_length == 0:
            self._ch_history.clear()
