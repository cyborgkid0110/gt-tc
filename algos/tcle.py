"""
TCLE: Topology Control with Lifetime Extension.

Non-cooperative power-control game with energy-aware adaptation.
Each sensor adjusts transmit power to balance connectivity (measured by
algebraic connectivity λ₂ of the graph Laplacian) against an integral-based
unwillingness cost that penalises energy-depleted nodes more heavily.

Key mechanisms:
  - Block-partitioned strategy set: κ_i = f(E_i), low energy → finer steps.
  - Wait-time priority: low-energy sensors adapt first.
  - Event-triggered reconstruction: topology is re-optimised when any sensor's
    unwillingness crosses one of K discrete levels.

Note on connectivity: the paper assumes bidirectional links, but this project
treats unidirectional links as valid. Algebraic connectivity is computed on the
undirected version of the graph (union of forward and reverse edges).

Note on energy model: the paper does not model per-round energy dissipation.
This implementation adds a maintenance phase using the shared NetworkModel
energy model (role='CM', clustering=False), same as DIA-MIA.

Reference: Xu et al., "Topology Control with Lifetime Extension for WSNs",
IEEE Trans. Vehicular Technology, 2016.
"""

import math
import random
import numpy as np
import yaml
import networkx as nx
from scipy import integrate

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot, tx_power_plot


class TCLE(BaseAlgorithm):
    """Topology Control with Lifetime Extension algorithm."""
    family = 'topology'

    def __init__(self, net: NetworkModel, config_path: str = 'config/tcle.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Algebraic connectivity threshold
        self.epsilon: float = cfg['epsilon']

        # Unwillingness / pricing
        self._pricing_name: str = cfg['pricing']
        self.mu: float = cfg['mu']

        # Block partition & event-triggered reconstruction
        self.K: int = cfg['kappa_levels']

        # Wait time priority
        self.tau: float = cfg['tau']
        self.sigma_max: float = cfg['sigma_max']

        # Flag: topology needs (re-)adaptation
        self._needs_adapt: bool = True

        # Previous unwillingness level per sensor (for event detection)
        self._prev_unwillingness: dict[int, int] = {s.id: 0 for s in net.sensors}

        # Safety limit for adaptation convergence
        self._max_adapt_iter: int = 300_000

        # Build discretised power levels (descending: p_max, p_max-δ, ..., p_min)
        self._power_levels: list[float] = []
        p = net.p_max
        while p >= net.p_min:
            self._power_levels.append(round(p, 12))
            p = round(p - net.p_step, 12)
        if self._power_levels[-1] > net.p_min:
            self._power_levels.append(net.p_min)
        self._num_levels = len(self._power_levels)

    # ------------------------------------------------------------------ #
    #  Pricing functions                                                   #
    # ------------------------------------------------------------------ #

    def _pricing_fn(self, x: float) -> float:
        if self._pricing_name == 'linear':
            return x
        elif self._pricing_name == 'quadratic':
            return x * x
        else:   # exponential (same as GT2)
            return np.exp(x / 10)

    # ------------------------------------------------------------------ #
    #  Single round                                                        #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        net = self.net

        if self._needs_adapt:
            self._initialize_topology()
            self._adapt()
            self._needs_adapt = False

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())
            tx_power_plot(net.to_network_dict(), net.to_node_dict())

        # Energy deduction
        self._maintenance()

        # Event-triggered reconstruction check
        if self._check_reconstruction_trigger():
            self._needs_adapt = True

        print(f'TCLE Round {self.t}: Dead={self.dead_nodes}/{net.num_nodes}')

        if self.dead_nodes >= net.num_nodes:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Topology initialisation                                             #
    # ------------------------------------------------------------------ #

    def _initialize_topology(self) -> None:
        """Set all alive nodes to p_max and build g_max."""
        net = self.net

        net.edges = np.zeros((net.num_nodes, net.num_nodes), dtype=int)
        for s in net.sensors:
            s.neighbors = []

        for s in net.sensors:
            if s.is_alive:
                s.power = net.p_max
                s.rc = net.calc_comm_range(net.p_max)

        net.discover_neighbors()

        alive = sum(1 for s in net.sensors if s.is_alive)
        lambda2 = self._compute_algebraic_connectivity()
        print(f'TCLE: initialised g_max ({alive} alive, λ₂={lambda2:.4f})')

    # ------------------------------------------------------------------ #
    #  Algebraic connectivity                                              #
    # ------------------------------------------------------------------ #

    def _compute_algebraic_connectivity(self) -> float:
        """Compute Fiedler value λ₂ on the undirected condensation of net.edges.

        Connectivity is validated on the directed graph first: the network
        must be strongly connected (every node can reach every other via
        directed multi-hop paths). Then λ₂ is the second-smallest eigenvalue
        of the Laplacian of the undirected version (union of edges).

        The Fiedler value is obtained from a dense eigensolve
        (``numpy.linalg.eigvalsh``) rather than ``nx.algebraic_connectivity``.
        For a few-hundred-node graph the dense solve is ~30-50× faster than
        networkx's iterative tracemin solver and returns the identical value
        (agreement to ~1e-13); this matters because ``_adapt`` calls this
        method hundreds of times per topology reconstruction.
        """
        net = self.net

        # Directed adjacency restricted to alive nodes.
        alive_idx = [i for i in range(net.num_nodes) if net.sensors[i].is_alive]
        if len(alive_idx) < 2:
            return 0.0
        sub = net.edges[np.ix_(alive_idx, alive_idx)]

        # Strong-connectivity gate on the directed sub-graph.
        D = nx.from_numpy_array(sub, create_using=nx.DiGraph)
        if not nx.is_strongly_connected(D):
            return 0.0

        # λ₂ on the undirected union of directed edges.
        A = ((sub + sub.T) > 0).astype(float)
        L = np.diag(A.sum(axis=1)) - A
        eigvals = np.linalg.eigvalsh(L)
        return float(eigvals[1])

    # ------------------------------------------------------------------ #
    #  Unwillingness function                                              #
    # ------------------------------------------------------------------ #

    def _compute_unwillingness(self, sensor: Sensor, power: float) -> float:
        """Integral-based unwillingness cost c_i(p_i, E_i).

        Structure matches GT2's calc_energy_cost() with configurable pricing.
        """
        T = 1.0
        lower = sensor.e0 - sensor.e_res
        upper = lower + power * T
        cost, _ = integrate.quad(self._pricing_fn, lower, upper)
        return cost / self.mu

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _utility_at_current_state(self, sensor: Sensor, power: float) -> float:
        """u_i = φ(λ₂ > ε) - c_i(p_i, E_i).

        Assumes net.edges already reflects the trial topology.
        """
        lambda2 = self._compute_algebraic_connectivity()
        phi = 1.0 if lambda2 > self.epsilon else 0.0
        c_i = self._compute_unwillingness(sensor, power)
        return phi - c_i

    # ------------------------------------------------------------------ #
    #  Block partition helpers                                             #
    # ------------------------------------------------------------------ #

    def _compute_kappa(self, sensor: Sensor) -> int:
        """Block size κ_i: inversely proportional to residual energy.

        Low energy → small κ → smaller blocks → finer-grained reduction.
        High energy → large κ → larger blocks → coarser steps.
        """
        ratio = sensor.e_res / sensor.e0 if sensor.e0 > 0 else 0.0
        kappa = max(1, math.ceil(self.K * ratio))
        return kappa

    def _get_current_block(self, sensor: Sensor, kappa: int) -> list[float]:
        """Return the candidate power levels for one adaptation step: the κ
        levels immediately below the sensor's current power.

        ``_power_levels`` is sorted high→low, so these are the next κ
        lower-power strategies. κ acts as the descent granularity — a small κ
        (low-energy node) takes fine, near single-step reductions; a large κ
        (high-energy node) may drop several levels at once. Across successive
        adaptation passes a node can descend the full ladder down to p_min.

        (Previously the ladder was carved into *fixed* blocks of size κ and the
        node was confined to the block holding its current power; once it
        reached a block's lower boundary — index 9 at full-energy κ=10 — it
        could never cross into the next block, so every node froze ~κ steps
        below p_max and the topology stayed near maximum power.)
        """
        # Find the index of current power (or nearest)
        current_idx = None
        for idx, p in enumerate(self._power_levels):
            if abs(p - sensor.power) < 1e-8:
                current_idx = idx
                break

        if current_idx is None:
            # Snap to nearest
            diffs = [abs(p - sensor.power) for p in self._power_levels]
            current_idx = diffs.index(min(diffs))

        # Candidate strategies: the next κ lower-power levels (sliding window).
        block_start = current_idx + 1
        block_end = min(block_start + kappa, self._num_levels)

        return self._power_levels[block_start:block_end]

    # ------------------------------------------------------------------ #
    #  Adaptation: block-partitioned with wait-time priority               #
    # ------------------------------------------------------------------ #

    def _adapt(self) -> None:
        """Block-partitioned power reduction with energy-priority ordering.

        Fast path: a trial power that loses no outgoing link leaves the graph
        topology identical, so the benefit φ = 1[λ₂ > ε] is unchanged and the
        145 ms Fiedler-value eigensolve is skipped — the utility delta reduces
        to the closed-form unwillingness change. λ₂ is recomputed only for the
        rare link-losing trials. The current-state φ is cached and invalidated
        only when an accepted move actually prunes a link.
        """
        net = self.net
        total_steps = 0

        alive = [s for s in net.sensors if s.is_alive]

        # Sort by wait time: t_w = τ * E_i + σ (lower energy → lower t_w → first)
        adapt_order = sorted(
            alive,
            key=lambda s: self.tau * s.e_res + random.uniform(0, self.sigma_max),
        )

        # Cache of the current graph's benefit indicator φ = 1[λ₂ > ε].
        # None ⇒ stale; recomputed lazily. Invalidated whenever an accepted
        # move prunes a link (the only thing that can change the topology).
        cached_phi: float | None = None

        def current_phi() -> float:
            nonlocal cached_phi
            if cached_phi is None:
                lambda2 = self._compute_algebraic_connectivity()
                cached_phi = 1.0 if lambda2 > self.epsilon else 0.0
            return cached_phi

        converged = False
        while not converged and total_steps < self._max_adapt_iter:
            converged = True

            for sensor in adapt_order:
                if not sensor.is_alive:
                    continue
                if sensor.power <= net.p_min:
                    continue

                kappa = self._compute_kappa(sensor)
                block = self._get_current_block(sensor, kappa)

                phi_cur = current_phi()
                current_util = phi_cur - self._compute_unwillingness(
                    sensor, sensor.power)

                # Try each lower power in the block
                best_power = sensor.power
                best_util = current_util

                for trial_power in block:
                    if trial_power >= sensor.power:
                        continue    # only try lower levels

                    trial_rc = net.calc_comm_range(trial_power)

                    # Identify links that would be lost
                    links_lost = [nb for nb in sensor.neighbors
                                  if sensor.distance_to(nb) > trial_rc]

                    if not links_lost:
                        # FAST PATH: topology unchanged ⇒ φ unchanged.
                        # u_i = φ - c_i, so only the unwillingness term moves.
                        trial_util = phi_cur - self._compute_unwillingness(
                            sensor, trial_power)
                    else:
                        # SLOW PATH: a link is lost ⇒ φ may change. Apply the
                        # change temporarily and recompute λ₂ on the trial graph.
                        old_neighbors = sensor.neighbors[:]
                        for nb in links_lost:
                            sensor.remove_neighbor(nb)
                            net.edges[sensor.id, nb.id] = 0

                        old_power = sensor.power
                        old_rc = sensor.rc
                        sensor.power = trial_power
                        sensor.rc = trial_rc

                        trial_util = self._utility_at_current_state(
                            sensor, trial_power)

                        # Restore original state for next trial
                        sensor.power = old_power
                        sensor.rc = old_rc
                        sensor.neighbors = old_neighbors
                        for nb in links_lost:
                            sensor.add_neighbor(nb)
                            net.edges[sensor.id, nb.id] = 1

                    if trial_util > best_util:
                        best_util = trial_util
                        best_power = trial_power

                # Apply best power found
                if best_power < sensor.power:
                    new_rc = net.calc_comm_range(best_power)
                    # Remove out-of-range links
                    pruned = False
                    for nb in sensor.neighbors[:]:
                        if sensor.distance_to(nb) > new_rc:
                            sensor.remove_neighbor(nb)
                            net.edges[sensor.id, nb.id] = 0
                            pruned = True
                    sensor.power = best_power
                    sensor.rc = new_rc
                    converged = False
                    if pruned:
                        # Topology changed ⇒ cached φ is stale.
                        cached_phi = None

                total_steps += 1

        lambda2 = self._compute_algebraic_connectivity()
        print(f'TCLE adapted in {total_steps} steps '
              f'(converged={converged}, λ₂={lambda2:.4f})')

    # ------------------------------------------------------------------ #
    #  Event-triggered reconstruction                                      #
    # ------------------------------------------------------------------ #

    def _check_reconstruction_trigger(self) -> bool:
        """Check if any sensor's unwillingness crossed a K-level boundary."""
        triggered = False

        for s in self.net.sensors:
            if not s.is_alive:
                continue

            c_i = self._compute_unwillingness(s, s.power)
            # Discretise into K levels
            level = min(int(c_i * self.K), self.K)
            prev = self._prev_unwillingness.get(s.id, 0)

            if level > prev:
                triggered = True

            self._prev_unwillingness[s.id] = level

        return triggered

    # ------------------------------------------------------------------ #
    #  Maintenance: energy deduction                                       #
    # ------------------------------------------------------------------ #

    def _maintenance(self) -> None:
        """Deduct one round of energy using routing-based per-hop TX cost."""
        net = self.net

        routing_tree = net.build_routing_tree()
        self._routing_tree = routing_tree
        costs = net.compute_maintenance_costs(routing_tree)

        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.id in costs:
                s.c_cm = costs[s.id]
                s.e_res -= s.c_cm
                if s.e_res <= 0:
                    self._track_death(s)
                    self._needs_adapt = True
