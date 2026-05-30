"""
SCA-Lévy: Sine-Cosine Algorithm with Lévy Mutation for WSN clustering.

Per round:
  1. Build a high-energy candidate pool and target CH count k_opt = round(N_alive * p).
  2. Initialise a population of m candidate CH groupings (each is k_opt 2-D positions).
  3. Iterate SCA position update with sinusoidal step-factor r1, snap each position to
     the nearest candidate sensor to keep selections on real nodes. After each SCA step,
     apply Lévy mutation to below-average individuals to escape local optima.
  4. Accept the best grouping (minimum intra-cluster distance-variance fitness) as CHs.
  5. Cluster formation (nearest CH), intra-cluster multi-hop maintenance, and multi-hop
     CH→CH→BS forwarding.

Notes (see PLAN.md):
  - Energy model is `NetworkModel`'s shared layer — not the paper's two-threshold radio.
  - Tent chaotic initialisation is not used (node positions come from `main.py`).
  - Intra-cluster, CH→CH and CH→BS are all multi-hop.
  - Project's directed-edge connectivity convention applies.
  - Eq. (16) relay-node design is implemented as a reference helper but not wired
    into the main maintenance loop.

Reference: Guo, X., Ye, Y., Li, L., Wu, R., & Sun, X. (2023).
           WSN Clustering Routing Algorithm Combining Sine Cosine Algorithm and
           Lévy Mutation. IEEE Access, 11, 22654–22663.
"""

import math
import random
import yaml
import numpy as np

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot
import graph


class SCALEVY(BaseAlgorithm):
    """SCA-Lévy clustering routing algorithm."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/sca_levy.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # SCA / Lévy
        self._m: int = int(cfg['population_size'])
        self._T: int = int(cfg['max_sca_iter'])
        self._a: float = float(cfg['sca_a'])
        self._b: float = float(cfg['sca_b'])
        self._levy_beta: float = float(cfg['levy_beta'])

        # Cluster sizing
        self._p_ch: float = float(cfg['p_ch_fraction'])
        self._high_e_frac: float = float(cfg['high_energy_fraction'])

        # Mantegna sigma_u (pre-computed — β fixed)
        b = self._levy_beta
        num = math.gamma(1 + b) * math.sin(math.pi * b / 2)
        den = math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2)
        self._sigma_u = (num / den) ** (1.0 / b)

        # Search bounds for SCA position update (deployment area)
        self._lb = -float(net.area)
        self._ub = float(net.area)

    # ------------------------------------------------------------------ #
    #  Round                                                               #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        net = self.net
        net.reset_round()

        # ---- reset per-round sensor state ---------------------------
        for s in net.sensors:
            s.is_ch = False
            s.ch_neighbors = []
            if s.is_alive:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)

        # ---- neighbour discovery ------------------------------------
        net.discover_neighbors()

        # ---- Phase 1: CH election via SCA-Lévy ----------------------
        alive = [s for s in net.sensors if s.is_alive]
        if not alive:
            return False

        k_opt = max(1, round(len(alive) * self._p_ch))
        candidates = self._build_candidate_set(alive)
        if len(candidates) < k_opt:
            candidates = alive  # degenerate: not enough high-energy nodes

        best_grouping = self._sca_levy_optimize(candidates, k_opt, alive)
        for s in best_grouping:
            s.is_ch = True

        print(f'SCA-LEVY Round {self.t}: CHs={len(best_grouping)}, '
              f'Dead={self.dead_nodes}/{net.num_nodes}')

        # ---- Phase 2: cluster formation -----------------------------
        self._cluster_formation()
        self._filter_neighbours()
        self._connect_unaffiliated()
        self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # ---- Phase 3: build layered batches + CH→BS routes ----------
        mod_edges = net.create_cluster_subgraph()
        mod_net_dict = net.to_network_dict(edges=mod_edges)
        node_dict = net.to_node_dict()
        G = graph.build_graph(mod_net_dict['vertices'], mod_net_dict['edges'])
        layered_batches = graph.divide_network_by_clusters(G, node_dict)

        ch_routes = self._build_ch_to_bs_routes(best_grouping)

        # ---- Phase 4: maintenance (energy deduction) ----------------
        self._maintenance(layered_batches, ch_routes)

        if self.dead_nodes >= net.num_nodes:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Candidate pool                                                      #
    # ------------------------------------------------------------------ #

    def _build_candidate_set(self, alive: list[Sensor]) -> list[Sensor]:
        """Top `high_energy_fraction` of alive nodes by residual energy."""
        if not alive:
            return []
        sorted_alive = sorted(alive, key=lambda s: s.e_res, reverse=True)
        n = max(1, int(len(sorted_alive) * self._high_e_frac))
        return sorted_alive[:n]

    # ------------------------------------------------------------------ #
    #  Phase 1: SCA-Lévy optimisation                                      #
    # ------------------------------------------------------------------ #

    def _sca_levy_optimize(self,
                           candidates: list[Sensor],
                           k_opt: int,
                           alive: list[Sensor]) -> list[Sensor]:
        """Run the population-based SCA-Lévy loop.

        Each particle is an array of shape (k_opt, 2) of continuous 2-D positions.
        After every position update each coordinate pair is snapped to its nearest
        candidate sensor; duplicates are resolved by taking the next-nearest.

        Fitness is evaluated over the full `alive` set so the objective matches
        the paper (Eq. 15 sums over N_alive − k_opt non-head members), not just
        the high-energy candidate subset.
        """
        cand_pos = np.array([[s.x, s.y] for s in candidates], dtype=float)
        alive_xy = np.array([[s.x, s.y] for s in alive], dtype=float)

        # --- initialise population: random k_opt picks from candidates ---
        population = np.empty((self._m, k_opt, 2), dtype=float)
        for i in range(self._m):
            idx = np.random.choice(len(candidates), size=k_opt, replace=False)
            population[i] = cand_pos[idx]

        snapped = self._snap_population(population, cand_pos)
        fitnesses = np.array([self._fitness(g, alive_xy)
                              for g in snapped])

        best_idx = int(np.argmin(fitnesses))
        best_grouping_pos = snapped[best_idx].copy()
        best_f = float(fitnesses[best_idx])

        # --- main SCA-Lévy loop ---
        for t in range(1, self._T + 1):
            r1 = self._a * math.sin(math.pi / 2 * (1 - t / self._T)) + self._b

            for i in range(self._m):
                for j in range(k_opt):
                    for d in range(2):
                        r2 = random.uniform(0.0, 2 * math.pi)
                        r3 = random.uniform(0.0, 2.0)
                        r4 = random.random()
                        delta = abs(r3 * best_grouping_pos[j, d]
                                    - population[i, j, d])
                        if r4 < 0.5:
                            step = r1 * math.sin(r2) * delta
                        else:
                            step = r1 * math.cos(r2) * delta
                        population[i, j, d] += step

                # clip to deployment area
                np.clip(population[i], self._lb, self._ub, out=population[i])

            snapped = self._snap_population(population, cand_pos)
            fitnesses = np.array([self._fitness(g, alive_xy)
                                  for g in snapped])

            # Paper applies Lévy mutation to individuals whose fitness is below
            # the population average. We minimise f, so "below average" means
            # f_i > mean (worse diversity seekers).
            mean_f = float(np.mean(fitnesses))
            for i in range(self._m):
                if fitnesses[i] <= mean_f:
                    continue
                for j in range(k_opt):
                    for d in range(2):
                        ls = self._levy_step()
                        delta = abs(best_grouping_pos[j, d]
                                    - population[i, j, d])
                        population[i, j, d] += ls * delta
                np.clip(population[i], self._lb, self._ub, out=population[i])

            snapped = self._snap_population(population, cand_pos)
            fitnesses = np.array([self._fitness(g, alive_xy)
                                  for g in snapped])

            idx_best = int(np.argmin(fitnesses))
            if fitnesses[idx_best] < best_f:
                best_f = float(fitnesses[idx_best])
                best_grouping_pos = snapped[idx_best].copy()

        # --- decode best grouping to Sensor objects ---
        return self._decode_to_sensors(best_grouping_pos, candidates, cand_pos)

    # ------------------------------------------------------------------ #
    #  Helpers: snap to candidates / fitness / Lévy step                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _snap_one(positions: np.ndarray,
                  cand_pos: np.ndarray) -> np.ndarray:
        """Snap each of k positions to the nearest distinct candidate position.

        Greedy: process heads in order; for each, pick the nearest candidate
        not already chosen.
        """
        k = positions.shape[0]
        used: set[int] = set()
        out = np.empty_like(positions)
        for j in range(k):
            diff = cand_pos - positions[j]
            d2 = (diff * diff).sum(axis=1)
            order = np.argsort(d2)
            for idx in order:
                if int(idx) not in used:
                    used.add(int(idx))
                    out[j] = cand_pos[idx]
                    break
        return out

    def _snap_population(self,
                         population: np.ndarray,
                         cand_pos: np.ndarray) -> np.ndarray:
        out = np.empty_like(population)
        for i in range(population.shape[0]):
            out[i] = self._snap_one(population[i], cand_pos)
        return out

    @staticmethod
    def _fitness(grouping_pos: np.ndarray,
                 alive_xy: np.ndarray) -> float:
        """Intra-cluster distance-variance fitness (paper Section 5, Eq. 15).

        grouping_pos: (k_opt, 2) snapped CH positions.
        alive_xy:     (N_alive, 2) positions of every alive node.

        Members = all alive nodes not located exactly at a head position. Each
        member is assigned to the nearest head by Euclidean distance, then
        sum_c = Σ d(member, CH_c)^2 per cluster. Fitness = Σ |sum_c − D/k_opt|.
        """
        k_opt = grouping_pos.shape[0]
        diff_all = alive_xy[:, None, :] - grouping_pos[None, :, :]
        d2_all = (diff_all * diff_all).sum(axis=2)        # (N_alive, k_opt)

        # Exclude heads (exact positional match on any head)
        is_head = (d2_all == 0.0).any(axis=1)
        d2 = d2_all[~is_head]
        if d2.shape[0] == 0:
            return 0.0

        assign = np.argmin(d2, axis=1)
        cluster_sums = np.zeros(k_opt, dtype=float)
        for i, ci in enumerate(assign):
            cluster_sums[int(ci)] += float(d2[i, ci])

        mean = cluster_sums.mean()
        return float(np.abs(cluster_sums - mean).sum())

    def _levy_step(self) -> float:
        """Mantegna Lévy step."""
        u = random.gauss(0.0, self._sigma_u)
        v = random.gauss(0.0, 1.0)
        if v == 0.0:
            return 0.0
        return u / (abs(v) ** (1.0 / self._levy_beta))

    @staticmethod
    def _decode_to_sensors(grouping_pos: np.ndarray,
                           candidates: list[Sensor],
                           cand_pos: np.ndarray) -> list[Sensor]:
        """Map snapped 2-D positions back to Sensor objects (no duplicates)."""
        chosen: list[Sensor] = []
        used: set[int] = set()
        for j in range(grouping_pos.shape[0]):
            diff = cand_pos - grouping_pos[j]
            d2 = (diff * diff).sum(axis=1)
            for idx in np.argsort(d2):
                if int(idx) not in used:
                    used.add(int(idx))
                    chosen.append(candidates[int(idx)])
                    break
        return chosen

    # ------------------------------------------------------------------ #
    #  Phase 2: cluster formation (bounded range, same pattern as LEACH)  #
    # ------------------------------------------------------------------ #

    def _cluster_formation(self):
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

    def _cleanup_cross_cluster_edges(self):
        net = self.net
        for s in net.sensors:
            for nb in s.neighbors[:]:
                if (s.ch_belong is not None and nb.ch_belong is not None
                        and s.ch_belong is not nb.ch_belong):
                    net.disconnect(s, nb)

    # ------------------------------------------------------------------ #
    #  Phase 3: multi-hop CH→BS routing (directed-graph BFS)               #
    # ------------------------------------------------------------------ #

    def _build_ch_to_bs_routes(self,
                               chs: list[Sensor]) -> dict[int, list[Sensor]]:
        """Return {ch_id: [hop1, hop2, ..., sink_ch]} — the CH chain each CH uses
        to reach the BS. The final hop transmits to the BS at the origin.

        Edges in the CH-CH graph: A→B if B ∈ A.ch_neighbors AND B is closer to
        the BS than A (monotone progress toward the sink). This avoids cycles and
        matches the paper's intent of forwarding toward the BS.
        """
        routes: dict[int, list[Sensor]] = {}
        alive_chs = [c for c in chs if c.is_alive]
        if not alive_chs:
            return routes

        def d_bs(s: Sensor) -> float:
            return math.hypot(s.x, s.y)

        # Adjacency: monotone progress toward BS via existing ch_neighbors
        adj: dict[int, list[Sensor]] = {c.id: [] for c in alive_chs}
        for c in alive_chs:
            d_c = d_bs(c)
            for nb in c.ch_neighbors:
                if nb.is_alive and nb.is_ch and d_bs(nb) < d_c:
                    adj[c.id].append(nb)

        # For each CH, pick the next hop that minimises remaining distance to BS.
        # Build the chain greedily; fall back to direct CH→BS if no CH hop exists.
        for c in alive_chs:
            chain: list[Sensor] = []
            cur = c
            visited = {cur.id}
            while True:
                hops = adj[cur.id]
                if not hops:
                    break
                # pick the CH hop closest to BS
                nxt = min(hops, key=d_bs)
                if nxt.id in visited:
                    break
                chain.append(nxt)
                visited.add(nxt.id)
                cur = nxt
            routes[c.id] = chain  # may be empty → direct CH→BS
        return routes

    # ------------------------------------------------------------------ #
    #  Phase 4: maintenance                                                #
    # ------------------------------------------------------------------ #

    def _maintenance(self,
                     layered_batches: dict,
                     ch_routes: dict[int, list[Sensor]]):
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

    # ------------------------------------------------------------------ #
    #  Reference: Eq. (16) relay-node selection (not in main flow)         #
    # ------------------------------------------------------------------ #

    def _select_relay_for_ch(self, ch: Sensor) -> Sensor | None:
        """Reference implementation of the paper's relay-node criterion (Eq. 16).

        A candidate relay R must satisfy:
          E_R > 0
          d(CH,R)^2 + d(R,BS)^2 < d(CH,BS)^2
          d(CH,R) < (1/√2) · d(CH,BS)
          d(R,BS) < (1/√2) · d(CH,BS)

        Among the feasible set, pick R maximising E_R / (d(CH,R)^2 + d(R,BS)^2).

        This helper is NOT called from `_maintenance` — multi-hop CH→CH routing
        already provides the same energy-saving function through the standard
        network topology. Kept here per PLAN.md as a reference.
        """
        net = self.net
        d_ch_bs = math.hypot(ch.x, ch.y)
        if d_ch_bs == 0.0:
            return None
        thresh = d_ch_bs / math.sqrt(2.0)
        d_ch_bs2 = d_ch_bs ** 2

        best = None
        best_ratio = -math.inf
        for s in net.sensors:
            if s is ch or not s.is_alive or s.is_ch:
                continue
            d_ch_r = ch.distance_to(s)
            d_r_bs = math.hypot(s.x, s.y)
            if d_ch_r >= thresh or d_r_bs >= thresh:
                continue
            d_sum2 = d_ch_r ** 2 + d_r_bs ** 2
            if d_sum2 >= d_ch_bs2 or d_sum2 == 0.0:
                continue
            ratio = s.e_res / d_sum2
            if ratio > best_ratio:
                best_ratio = ratio
                best = s
        return best
