"""
GTFR: Game Theory-Based Fuzzy Routing Protocol.

Three-phase protocol:
  Phase 1 — Fuzzy C-Means (FCM) clustering on node positions (run once, static).
  Phase 2 — Game-theoretic CH selection:
             2a. Tentative CHs via mixed Nash equilibrium (Eq. 11)
             2b. Final CH per cluster via fitness function (Eq. 15)
  Phase 3 — Data communication: single-hop CM→CH, CH aggregates and forwards to BS.

Reference: Gangwar et al., IEEE Sensors Journal, Vol. 24, No. 6, March 2024.
"""

import random
import numpy as np
import yaml

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot


class GTFR(BaseAlgorithm):
    """Game Theory-Based Fuzzy Routing protocol."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/gtfr.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # FCM parameters
        self.fcm_cluster_fraction = cfg['fcm_cluster_fraction']
        self.fcm_fuzziness = cfg['fcm_fuzziness']
        self.fcm_beta = cfg['fcm_beta']
        self.fcm_max_iter = cfg['fcm_max_iter']

        # Psi weights (Eq. 7)
        self.psi_alpha = cfg['psi_alpha']
        self.psi_beta = cfg['psi_beta']
        self.psi_gamma = cfg['psi_gamma']
        self.psi_delta = cfg['psi_delta']

        # Fitness function lookup tables (Eq. 15)
        self.reward_table = cfg['reward_table']
        self.penalty_nh_table = cfg['penalty_nh_table']
        self.penalty_iacd_table = cfg['penalty_iacd_table']

        # CH history: sensor_id -> number of times elected as final CH
        self._ch_count: dict[int, int] = {s.id: 0 for s in net.sensors}

        # Run FCM once — positions are static, result is the same every round
        self._fcm_clusters, self._node_cluster = self._run_fcm()

        # Pre-compute IACD values (static: depend only on positions)
        self._iacd: dict[int, float] = self._compute_iacd()

        num_clusters = len(self._fcm_clusters)
        sizes = [len(v) for v in self._fcm_clusters.values()]
        print(f'GTFR init: FCM produced {num_clusters} clusters, '
              f'sizes: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}')

    # ------------------------------------------------------------------ #
    #  Single round                                                        #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        net = self.net

        # ---- reset --------------------------------------------------
        net.reset_round()

        for s in net.sensors:
            s.is_ch = False
            s.ch_neighbors = []
            if s.is_alive:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)

        # ---- Phase 1: neighbour discovery (needed for ND in psi) ----
        net.discover_neighbors()

        # ---- Phase 2a: Tentative CH selection -----------------------
        tchs = self._select_tentative_chs()

        # ---- Phase 2b: Final CH selection ---------------------------
        final_chs = self._select_final_chs(tchs)

        print(f'GTFR Round {self.t}: CHs={len(final_chs)}, '
              f'Dead={self.dead_nodes}/{net.num_nodes}')

        if len(final_chs) > 0:
            # ---- Phase 3a: cluster formation ------------------------
            self._cluster_formation()
            self._filter_neighbours()
            self._connect_unaffiliated()
            self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # ---- Phase 3b: energy deduction (steady-state) --------------
        self._steady_state()

        # ---- update CH rotation history -----------------------------
        self._update_ch_history()

        if self.dead_nodes >= net.num_nodes:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Phase 1: FCM clustering (runs once in __init__)                    #
    # ------------------------------------------------------------------ #

    def _run_fcm(self) -> tuple[dict[int, list[Sensor]], dict[int, int]]:
        """Run Fuzzy C-Means on node positions. Returns (clusters, node_cluster_map).

        clusters: cluster_id (int) -> list of Sensor objects
        node_cluster_map: sensor_id (int) -> cluster_id (int)
        """
        sensors = self.net.sensors
        n = len(sensors)
        c = max(1, int(self.fcm_cluster_fraction * n))
        m = self.fcm_fuzziness

        positions = np.array([[s.x, s.y] for s in sensors], dtype=float)

        # Random initialisation of membership matrix; rows sum to 1
        rng = np.random.default_rng(seed=0)
        mu = rng.random((n, c))
        mu = mu / mu.sum(axis=1, keepdims=True)

        of_prev = None

        for _ in range(self.fcm_max_iter):
            mu_m = mu ** m  # (n, c)

            # Update centroids (Eq. 2-3)
            weights = mu_m.sum(axis=0)  # (c,)
            centroids = (mu_m.T @ positions) / weights[:, None]  # (c, 2)

            # Compute squared distances: dist[i, j] = ||S_i - C_j||^2
            diff = positions[:, None, :] - centroids[None, :, :]  # (n, c, 2)
            dist = (diff ** 2).sum(axis=2)  # (n, c)

            # Objective function
            of = float((mu_m * dist).sum())

            # Check convergence
            if of_prev is not None and abs(of_prev - of) < self.fcm_beta:
                break
            of_prev = of

            # Update membership (Eq. 5)
            # mu[i,j] = 1 / sum_k (dist[i,j] / dist[i,k])^(1/(m-1))
            exp = 1.0 / (m - 1)
            new_mu = np.zeros((n, c), dtype=float)

            for i in range(n):
                # Check if node sits exactly on any centroid
                zero_mask = dist[i] == 0.0
                if zero_mask.any():
                    new_mu[i, zero_mask] = 1.0 / zero_mask.sum()
                    new_mu[i, ~zero_mask] = 0.0
                else:
                    ratios = dist[i, :, None] / dist[i, None, :]  # (c, c)
                    new_mu[i] = 1.0 / (ratios ** exp).sum(axis=1)

            mu = new_mu

        # Hard cluster assignment
        assignments = np.argmax(mu, axis=1)  # (n,)

        clusters: dict[int, list[Sensor]] = {j: [] for j in range(c)}
        node_cluster: dict[int, int] = {}
        for idx, s in enumerate(sensors):
            cid = int(assignments[idx])
            clusters[cid].append(s)
            node_cluster[s.id] = cid

        return clusters, node_cluster

    def _compute_iacd(self) -> dict[int, float]:
        """Pre-compute IACD for each node: avg distance to other members of same FCM cluster."""
        iacd: dict[int, float] = {}
        for members in self._fcm_clusters.values():
            for s in members:
                others = [o for o in members if o is not s]
                if others:
                    iacd[s.id] = sum(s.distance_to(o) for o in others) / len(others)
                else:
                    iacd[s.id] = 0.0
        return iacd

    # ------------------------------------------------------------------ #
    #  Phase 2a: Tentative CH selection                                   #
    # ------------------------------------------------------------------ #

    def _select_tentative_chs(self) -> list[Sensor]:
        """Mixed Nash Equilibrium TCH selection (Eq. 11).

        Returns list of tentative CH candidates.
        """
        net = self.net
        alive = [s for s in net.sensors if s.is_alive]

        # Gather per-node values for normalisation
        iacd_vals = [self._iacd[s.id] for s in alive]
        nd_vals = [len(s.neighbors) for s in alive]
        e_vals = [s.e_res for s in alive]
        nh_vals = [self._ch_count[s.id] / max(1, self.t) for s in alive]

        iacd_min, iacd_max = min(iacd_vals), max(iacd_vals)
        nd_min,   nd_max   = min(nd_vals),   max(nd_vals)
        e_min,    e_max    = min(e_vals),     max(e_vals)
        nh_min,   nh_max   = min(nh_vals),    max(nh_vals)

        def _norm_hi(val, vmin, vmax):
            """(max - val) / (max - min): higher val → lower score."""
            if vmax == vmin:
                return 0.0
            return (vmax - val) / (vmax - vmin)

        tchs = []

        for s in alive:
            nd = len(s.neighbors)
            if nd <= 1:
                # Cannot compute 1/(ND-1); skip
                continue

            c_ch = net.calc_node_cost(s, 'CH', clustering=True)
            c_cm = net.calc_node_cost(s, 'CM', clustering=True)
            s.c_ch = c_ch
            s.c_cm = c_cm

            # Penalty coefficient psi (Eq. 7)
            iacd_term = _norm_hi(self._iacd[s.id], iacd_min, iacd_max)
            nd_term   = _norm_hi(nd, nd_min, nd_max)
            e_term    = _norm_hi(s.e_res, e_min, e_max)
            nh_term   = _norm_hi(self._ch_count[s.id] / max(1, self.t), nh_min, nh_max)

            psi = (self.psi_alpha * iacd_term
                   + self.psi_beta  * nd_term
                   + self.psi_gamma * e_term
                   + self.psi_delta * nh_term)

            # NE equilibrium probability (Eq. 11)
            if psi == 0.0:
                p_i = 0.0
            elif psi * c_ch <= c_cm:
                p_i = 1.0
            else:
                base = (psi * c_ch - c_cm) / (psi * c_ch)
                base = max(0.0, min(1.0, base))  # clamp to [0,1]
                p_i = 1.0 - base ** (1.0 / (nd - 1))
                p_i = max(0.0, min(1.0, p_i))

            s.p_ch = p_i

            if p_i >= random.random():
                tchs.append(s)

        return tchs

    # ------------------------------------------------------------------ #
    #  Phase 2b: Final CH selection via fitness function                  #
    # ------------------------------------------------------------------ #

    def _lookup_reward(self, rer: float) -> float:
        for entry in self.reward_table:
            if entry['rer_min'] <= rer < entry['rer_max']:
                return float(entry['R'])
        return float(self.reward_table[-1]['R'])

    def _lookup_penalty_nh(self, nh_p: float) -> float:
        for entry in self.penalty_nh_table:
            if entry['nh_min'] <= nh_p < entry['nh_max']:
                return float(entry['P'])
        return float(self.penalty_nh_table[-1]['P'])

    def _lookup_penalty_iacd(self, iacd_norm: float) -> float:
        for entry in self.penalty_iacd_table:
            if entry['iacd_min'] <= iacd_norm < entry['iacd_max']:
                return float(entry['P'])
        return float(self.penalty_iacd_table[-1]['P'])

    def _select_final_chs(self, tchs: list[Sensor]) -> list[Sensor]:
        """Fitness function evaluation (Eq. 15). Elects one final CH per cluster.

        Returns list of elected CHs.
        """
        # Group TCHs by FCM cluster
        cluster_tchs: dict[int, list[Sensor]] = {}
        for s in tchs:
            cid = self._node_cluster[s.id]
            cluster_tchs.setdefault(cid, []).append(s)

        final_chs = []

        for cid, candidates in cluster_tchs.items():
            # IACD normalisation within this cluster's candidates
            iacd_vals = [self._iacd[s.id] for s in candidates]
            iacd_min = min(iacd_vals)
            iacd_max = max(iacd_vals)

            best_s = None
            best_fp = None

            for s in candidates:
                rer = (s.e_res / s.e0) * 100.0
                nh_p = self._ch_count[s.id] / max(1, self.t)

                if iacd_max == iacd_min:
                    iacd_norm = 0.0
                else:
                    iacd_norm = (self._iacd[s.id] - iacd_min) / (iacd_max - iacd_min)

                R = self._lookup_reward(rer)
                P_nh = self._lookup_penalty_nh(nh_p)
                P_iacd = self._lookup_penalty_iacd(iacd_norm)

                fp = R * rer - P_nh - P_iacd * iacd_norm  # Eq. 15

                if best_fp is None or fp > best_fp:
                    best_fp = fp
                    best_s = s

            if best_s is not None:
                best_s.is_ch = True
                final_chs.append(best_s)

        return final_chs

    # ------------------------------------------------------------------ #
    #  Phase 3a: Cluster formation (bounded range, same as LEACH/GT2)    #
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

        If a node reaches p_max without connecting, its power is set to 0.
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
    #  Phase 3b: Energy deduction (steady-state)                          #
    # ------------------------------------------------------------------ #

    def _steady_state(self):
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
    #  CH history                                                          #
    # ------------------------------------------------------------------ #

    def _update_ch_history(self):
        """Increment CH count for this round's elected CHs."""
        for s in self.net.sensors:
            if s.is_ch:
                self._ch_count[s.id] += 1
