"""
DIA-MIA: Topology Control Game algorithms.

Both algorithms model each node as a player in a non-cooperative power-control
game. Utility: u_i(p) = M * f_i(p) - p_i, where f_i(p) is the number of nodes
reachable from i via directed multi-hop paths and M = p_max is the benefit
multiplier. This is an Ordinal Potential Game; potential maximisers are
globally energy-efficient topologies.

  DIA (delta-Improvement Algorithm):
    Restrained better-response — each node decrements power by one step (delta)
    only if it strictly improves utility. Converges to the minmax energy-efficient
    topology (subset of PMST). O(n^2) convergence, Pareto-optimal, unique NE.

  MIA (Max-Improvement Algorithm):
    Greedy best-response — each node finds the power level that maximises its
    utility. Converges in O(n) steps (one pass) but suffers from first-mover
    advantage, producing unfair power distributions.

Note on energy model: the paper describes only topology design with no
energy-dissipation model. This implementation adds a maintenance phase after
topology convergence so that the algorithm can be benchmarked against other
protocols on network lifetime metrics. Energy is deducted using the shared
NetworkModel.calc_node_cost() interface (role='CM', clustering=False).

Note on connectivity: the paper assumes bidirectional links. This project treats
unidirectional links as valid connectivity (per project convention).

Reference: Komali, MacKenzie & Gilles, "Effect of Selfish Node Behavior on
Efficient Topology Control", IEEE Trans. Mobile Computing, 2008.
"""

import math
import yaml
from collections import deque

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot, tx_power_plot


class DIAMIA(BaseAlgorithm):
    """DIA / MIA Topology Control Game algorithm."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/dia_mia.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.mode: str = cfg['mode']   # 'DIA' or 'MIA'
        self.M: float = net.p_max      # benefit multiplier (Eq. 9)
        # LDIA k-hop limit: None → global DIA; positive int → LDIA (Eq. 16)
        self.dia_k: int | None = cfg.get('dia_k', None)

        # Flag: adapt topology before the next maintenance round
        self._needs_adapt: bool = True

        # Safety limit for DIA convergence loop
        # Theoretical bound: O(n * levels) where levels = (p_max-p_min)/p_step
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

        print(f'DIAMIA ({self.mode}) Round {self.t}: '
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
        import numpy as np
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
        print(f'DIAMIA ({self.mode}): initialised g_max '
              f'({alive_count} alive nodes)')

    # ------------------------------------------------------------------ #
    #  Adaptation dispatch                                                 #
    # ------------------------------------------------------------------ #

    def _adapt(self) -> None:
        if self.mode == 'MIA':
            self._adapt_mia()
        else:
            self._adapt_dia()

    # ------------------------------------------------------------------ #
    #  DIA: delta-Improvement Algorithm                                   #
    # ------------------------------------------------------------------ #

    def _adapt_dia(self) -> None:
        """Restrained better-response: decrement power by one step if utility
        improves. Iterate passes until no node can improve (Nash equilibrium).
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
                    # FAST PATH: no link lost → reachability unchanged →
                    # utility strictly improves (M * f_i - new_power > M * f_i - old_power)
                    sensor.power = new_power
                    sensor.rc = new_rc
                    converged = False
                    total_steps += 1
                    continue

                # SLOW PATH: a link would be lost; evaluate utility change
                old_reach = self._reach(sensor)
                old_util = self.M * old_reach - sensor.power

                # Temporarily remove lost outgoing links
                for nb in links_lost:
                    sensor.remove_neighbor(nb)
                    net.edges[sensor.id, nb.id] = 0

                new_reach = self._reach(sensor)
                new_util = self.M * new_reach - new_power

                if new_util > old_util:
                    # Accept decrement
                    sensor.power = new_power
                    sensor.rc = new_rc
                    converged = False
                else:
                    # Reject: restore lost links
                    for nb in links_lost:
                        sensor.add_neighbor(nb)
                        net.edges[sensor.id, nb.id] = 1

                total_steps += 1

        variant = f'LDIA(k={self.dia_k})' if self.dia_k is not None else 'DIA'
        print(f'{variant} converged in {total_steps} steps '
              f'(converged={converged})')

    # ------------------------------------------------------------------ #
    #  MIA: Max-Improvement Algorithm                                     #
    # ------------------------------------------------------------------ #

    def _adapt_mia(self) -> None:
        """Greedy best-response: each node finds the power level that maximises
        its utility in a single pass (round-robin order).
        """
        net = self.net

        for sensor in net.sensors:
            if not sensor.is_alive:
                continue

            # Compute reachability at current power (p_max after initialisation)
            best_reach = self._compute_reachability(sensor)
            best_util = self.M * best_reach - sensor.power
            best_power = sensor.power

            # Descend from current power: accept lowest power preserving
            # or improving utility
            trial_power = sensor.power

            while trial_power > net.p_min:
                trial_power = max(
                    round(trial_power - net.p_step, 6), net.p_min)

                trial_rc = net.calc_comm_range(trial_power)

                # Links that would be lost at trial_power
                would_lose = [nb for nb in sensor.neighbors
                              if sensor.distance_to(nb) > trial_rc]

                if not would_lose:
                    # No link lost: reachability preserved, utility improves
                    best_power = trial_power
                    # best_reach unchanged
                    if trial_power == net.p_min:
                        break
                    continue

                # Some links would be lost: check reachability
                # Temporarily remove them
                for nb in would_lose:
                    sensor.remove_neighbor(nb)
                    net.edges[sensor.id, nb.id] = 0

                trial_reach = self._compute_reachability(sensor)
                trial_util = self.M * trial_reach - trial_power

                # Restore links
                for nb in would_lose:
                    sensor.add_neighbor(nb)
                    net.edges[sensor.id, nb.id] = 1

                if trial_util > best_util:
                    best_util = trial_util
                    best_power = trial_power
                    best_reach = trial_reach

                if trial_reach < best_reach:
                    # Further decrements will only reduce reachability more
                    break

                if trial_power == net.p_min:
                    break

            # Apply best power: update power and prune out-of-range links
            sensor.power = best_power
            net.update_comm_range(sensor)
            for nb in sensor.neighbors[:]:
                if sensor.distance_to(nb) > sensor.rc:
                    net.disconnect(sensor, nb)

        alive_count = sum(1 for s in net.sensors if s.is_alive)
        print(f'MIA adapted in 1 pass ({alive_count} nodes)')

    # ------------------------------------------------------------------ #
    #  Reachability via BFS                                                #
    # ------------------------------------------------------------------ #

    def _reach(self, sensor: Sensor) -> int:
        """Reachability dispatcher: global BFS or k-hop BFS depending on dia_k."""
        if self.dia_k is None:
            return self._compute_reachability(sensor)
        return self._compute_reachability_khop(sensor, self.dia_k)

    def _compute_reachability(self, sensor: Sensor) -> int:
        """Count nodes reachable from *sensor* via directed multi-hop paths.

        Traverses the directed neighbour graph (sensor.neighbors).
        Returns a count that includes the sensor itself.
        """
        visited: set[int] = {sensor.id}
        queue: deque[Sensor] = deque([sensor])

        while queue:
            current = queue.popleft()
            for nb in current.neighbors:
                if nb.id not in visited and nb.is_alive:
                    visited.add(nb.id)
                    queue.append(nb)

        return len(visited)

    def _compute_reachability_khop(self, sensor: Sensor, k: int) -> int:
        """Count nodes reachable from *sensor* within at most k hops (LDIA, Eq. 16).

        Uses BFS with a hop counter. Returns a count including the sensor itself.
        """
        visited: set[int] = {sensor.id}
        # queue entries: (node, hops_remaining)
        queue: deque[tuple[Sensor, int]] = deque([(sensor, k)])

        while queue:
            current, hops_left = queue.popleft()
            if hops_left == 0:
                continue
            for nb in current.neighbors:
                if nb.id not in visited and nb.is_alive:
                    visited.add(nb.id)
                    queue.append((nb, hops_left - 1))

        return len(visited)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _utility(self, sensor: Sensor) -> float:
        """u_i(p) = M * f_i(p) - p_i  (Eq. 9)."""
        return self.M * self._compute_reachability(sensor) - sensor.power

    def _omega(self, si: Sensor, sj: Sensor) -> float:
        """Minimum power for si to reach sj (inverse Friis model)."""
        d = si.distance_to(sj)
        return self.net.pth * (4 * math.pi * d / self.net.wave) ** 2

    # ------------------------------------------------------------------ #
    #  Maintenance: energy deduction                                       #
    # ------------------------------------------------------------------ #

    def _maintenance(self) -> None:
        """Deduct one round of energy from all alive nodes.

        Each DIA-MIA node acts as a sensing + transmitting node (CM role,
        no aggregation). Distance = sensor.rc (post-adaptation comm range).
        """
        net = self.net

        for s in net.sensors:
            if not s.is_alive:
                continue

            # CM role: sense + process + transmit (no aggregation)
            # clustering=False → d = s.rc (actual post-adaptation range)
            cost = net.calc_node_cost(s, 'CM', clustering=False)
            s.c_cm = cost
            s.e_res -= cost

            if s.e_res <= 0:
                self._track_death(s)
                # Topology changed: re-adapt before the next maintenance round
                self._needs_adapt = True
