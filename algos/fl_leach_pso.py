"""
FL-LEACH-PSO: Fuzzy Logic LEACH with Particle Swarm Optimization.

Two-phase protocol:
  Setup (once):  Hybrid PSO + K-Means clustering via Gap statistic.
  Steady (per round):
    1. PCH selection via Mamdani fuzzy logic (3 inputs, 27 rules)
    2. SCH selection via Mamdani fuzzy logic (2 inputs, 9 rules)
    3. Intracluster data communication: CM -> SCH -> PCH -> BS
    4. Energy deduction (maintenance)

Two-tier CH hierarchy:
  PCH (Primary CH) — receives aggregated data from SCH, forwards to BS.
  SCH (Secondary CH) — collects raw data from CMs, aggregates, forwards to PCH.

Note on energy model: the paper uses a different radio model. This implementation
uses the shared NetworkModel energy model for fair benchmarking. Both PCH and SCH
are treated as 'CH' role; CMs as 'CM' role. Transmission distances are set to the
actual target (SCH->PCH distance, PCH->BS distance, CM->SCH distance).

Reference: Gamal et al., "An Efficient Fuzzy Logic LEACH Technique-Based
Particle Swarm Optimization for Wireless Sensor Networks", Sensors, 2022.
"""

import math
import random
import yaml
import numpy as np

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot


class FLLEACHPSO(BaseAlgorithm):
    """FL-LEACH-PSO clustering protocol."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/fl_leach_pso.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # PSO parameters
        self._pso_particles: int = cfg['pso_particles']
        self._pso_max_iter: int = cfg['pso_max_iter']
        self._pso_w: float = cfg['pso_w']
        self._pso_c1: float = cfg['pso_c1']
        self._pso_c2: float = cfg['pso_c2']

        # K-Means
        self._km_max_iter: int = cfg['kmeans_max_iter']

        # Gap statistic
        self._gap_k_max: int = cfg['gap_k_max']
        self._gap_n_ref: int = cfg['gap_n_ref']

        # Max distances for fuzzy MF normalization (Eq. 12-13)
        self._max_energy: float = net.sensors[0].e0
        self._max_dist: float = math.hypot(net.area, net.area)

        # Fuzzy output discretization
        self._out_x = np.linspace(0, 100, 201)

        # PCH output MFs: 9 levels evenly spaced over [0, 100]
        self._pch_out_mfs = {
            'very_weak':     (0,    0,    12.5),
            'weak':          (0,    12.5, 25),
            'little_weak':   (12.5, 25,   37.5),
            'little_medium': (25,   37.5, 50),
            'medium':        (37.5, 50,   62.5),
            'high_medium':   (50,   62.5, 75),
            'little_strong': (62.5, 75,   87.5),
            'strong':        (75,   87.5, 100),
            'very_strong':   (87.5, 100,  100),
        }

        # SCH output MFs: 5 levels evenly spaced over [0, 100]
        self._sch_out_mfs = {
            'very_low':  (0,  0,  25),
            'low':       (0,  25, 50),
            'medium':    (25, 50, 75),
            'high':      (50, 75, 100),
            'very_high': (75, 100, 100),
        }

        # PCH rule table (Table 1): (energy, dist_center, dist_bs) -> output
        self._pch_rules = [
            ('low', 'distant', 'distant', 'very_weak'),
            ('low', 'distant', 'adequate', 'weak'),
            ('low', 'distant', 'near', 'little_weak'),
            ('low', 'adequate', 'distant', 'weak'),
            ('low', 'adequate', 'adequate', 'little_weak'),
            ('low', 'adequate', 'near', 'little_medium'),
            ('low', 'near', 'distant', 'little_weak'),
            ('low', 'near', 'adequate', 'little_medium'),
            ('low', 'near', 'near', 'medium'),
            ('medium', 'distant', 'distant', 'little_weak'),
            ('medium', 'distant', 'adequate', 'little_medium'),
            ('medium', 'distant', 'near', 'medium'),
            ('medium', 'adequate', 'distant', 'little_medium'),
            ('medium', 'adequate', 'adequate', 'medium'),
            ('medium', 'adequate', 'near', 'high_medium'),
            ('medium', 'near', 'distant', 'medium'),
            ('medium', 'near', 'adequate', 'high_medium'),
            ('medium', 'near', 'near', 'little_strong'),
            ('high', 'distant', 'distant', 'medium'),
            ('high', 'distant', 'adequate', 'high_medium'),
            ('high', 'distant', 'near', 'little_strong'),
            ('high', 'adequate', 'distant', 'high_medium'),
            ('high', 'adequate', 'adequate', 'little_strong'),
            ('high', 'adequate', 'near', 'strong'),
            ('high', 'near', 'distant', 'little_strong'),
            ('high', 'near', 'adequate', 'strong'),
            ('high', 'near', 'near', 'very_strong'),
        ]

        # SCH rule table (Table 2): (energy, dist_pch) -> output
        self._sch_rules = [
            ('low', 'distant', 'very_low'),
            ('medium', 'distant', 'low'),
            ('high', 'distant', 'medium'),
            ('low', 'adequate', 'low'),
            ('medium', 'adequate', 'medium'),
            ('high', 'adequate', 'medium'),
            ('low', 'near', 'medium'),
            ('medium', 'near', 'high'),
            ('high', 'near', 'very_high'),
        ]

        # Run PSO+K-Means clustering once (positions are static)
        positions = np.array([[s.x, s.y] for s in net.sensors], dtype=float)
        n_clusters = self._gap_statistic(positions,
                                         range(2, min(self._gap_k_max + 1,
                                                      len(positions))))
        assignments, centroids = self._pso_kmeans(positions, n_clusters)

        # Cache cluster assignments
        self._clusters: dict[int, list[Sensor]] = {j: [] for j in range(n_clusters)}
        self._node_cluster: dict[int, int] = {}
        self._centroids: dict[int, tuple[float, float]] = {}

        for idx, s in enumerate(net.sensors):
            cid = int(assignments[idx])
            self._clusters[cid].append(s)
            self._node_cluster[s.id] = cid

        for j in range(n_clusters):
            self._centroids[j] = (float(centroids[j, 0]),
                                  float(centroids[j, 1]))

        # Remove empty clusters
        empty = [k for k, v in self._clusters.items() if len(v) == 0]
        for k in empty:
            del self._clusters[k]
            del self._centroids[k]

        # Per-round PCH/SCH tracking
        self._pchs: dict[int, Sensor] = {}
        self._schs: dict[int, Sensor] = {}

        sizes = [len(v) for v in self._clusters.values()]
        print(f'FL-LEACH-PSO init: {len(self._clusters)} clusters, '
              f'sizes: min={min(sizes)} max={max(sizes)} '
              f'avg={sum(sizes)/len(sizes):.1f}')

    # ------------------------------------------------------------------ #
    #  Fuzzy logic primitives                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trimf(x: float, abc: tuple[float, float, float]) -> float:
        """Triangular membership function."""
        a, b, c = abc
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return (x - a) / (b - a) if b > a else 1.0
        return (c - x) / (c - b) if c > b else 1.0

    @staticmethod
    def _trapmf(x: float, abcd: tuple[float, float, float, float]) -> float:
        """Trapezoidal membership function."""
        a, b, c, d = abcd
        if x <= a or x >= d:
            return 0.0
        if x <= b:
            return (x - a) / (b - a) if b > a else 1.0
        if x <= c:
            return 1.0
        return (d - x) / (d - c) if d > c else 1.0

    def _energy_mf(self, x: float) -> dict[str, float]:
        """Evaluate energy membership functions (3 levels over [0, E0])."""
        e = self._max_energy
        return {
            'low':    self._trapmf(x, (0, 0, 0.15 * e, 0.35 * e)),
            'medium': self._trimf(x, (0.15 * e, 0.35 * e, 0.65 * e)),
            'high':   self._trapmf(x, (0.45 * e, 0.65 * e, e, e)),
        }

    def _distance_mf(self, x: float, max_d: float) -> dict[str, float]:
        """Evaluate distance membership functions (3 levels over [0, max_d])."""
        return {
            'near':     self._trapmf(x, (0, 0, 0.2 * max_d, 0.4 * max_d)),
            'adequate': self._trimf(x, (0.2 * max_d, 0.5 * max_d, 0.8 * max_d)),
            'distant':  self._trapmf(x, (0.6 * max_d, 0.8 * max_d, max_d, max_d)),
        }

    def _defuzzify_coa(self, aggregated: np.ndarray) -> float:
        """Center of Area defuzzification."""
        total = aggregated.sum()
        if total == 0:
            return 0.0
        return float(np.dot(self._out_x, aggregated) / total)

    # ------------------------------------------------------------------ #
    #  PCH / SCH fuzzy inference                                           #
    # ------------------------------------------------------------------ #

    def _fuzzy_pch_chance(self, energy: float, dist_center: float,
                          dist_bs: float) -> float:
        """Mamdani FIS for PCH selection chance (27 rules, 3 inputs)."""
        e_mf = self._energy_mf(energy)
        dc_mf = self._distance_mf(dist_center, self._max_dist)
        db_mf = self._distance_mf(dist_bs, self._max_dist)

        aggregated = np.zeros_like(self._out_x)

        for e_level, dc_level, db_level, out_level in self._pch_rules:
            strength = min(e_mf[e_level], dc_mf[dc_level], db_mf[db_level])
            if strength > 0:
                out_params = self._pch_out_mfs[out_level]
                mf_vals = np.array([self._trimf(x, out_params)
                                    for x in self._out_x])
                clipped = np.minimum(mf_vals, strength)
                aggregated = np.maximum(aggregated, clipped)

        return self._defuzzify_coa(aggregated)

    def _fuzzy_sch_chance(self, energy: float, dist_pch: float) -> float:
        """Mamdani FIS for SCH selection chance (9 rules, 2 inputs)."""
        e_mf = self._energy_mf(energy)
        dp_mf = self._distance_mf(dist_pch, self._max_dist)

        aggregated = np.zeros_like(self._out_x)

        for e_level, dp_level, out_level in self._sch_rules:
            strength = min(e_mf[e_level], dp_mf[dp_level])
            if strength > 0:
                out_params = self._sch_out_mfs[out_level]
                mf_vals = np.array([self._trimf(x, out_params)
                                    for x in self._out_x])
                clipped = np.minimum(mf_vals, strength)
                aggregated = np.maximum(aggregated, clipped)

        return self._defuzzify_coa(aggregated)

    # ------------------------------------------------------------------ #
    #  PCH / SCH selection                                                 #
    # ------------------------------------------------------------------ #

    def _select_pch(self) -> dict[int, Sensor]:
        """Select one PCH per cluster via fuzzy logic."""
        pchs: dict[int, Sensor] = {}

        for cid, members in self._clusters.items():
            alive = [s for s in members if s.is_alive]
            if not alive:
                continue

            cx, cy = self._centroids[cid]
            best_s = None
            best_chance = -1.0

            for s in alive:
                dist_center = math.hypot(s.x - cx, s.y - cy)
                dist_bs = math.hypot(s.x, s.y)
                chance = self._fuzzy_pch_chance(s.e_res, dist_center, dist_bs)

                # Tie-break by residual energy
                if (chance > best_chance or
                        (chance == best_chance and best_s is not None
                         and s.e_res > best_s.e_res)):
                    best_chance = chance
                    best_s = s

            if best_s is not None:
                best_s.is_ch = True
                pchs[cid] = best_s

        self._pchs = pchs
        return pchs

    def _select_sch(self) -> dict[int, Sensor]:
        """Select one SCH per cluster (excluding PCH) via fuzzy logic.

        Clusters with fewer than 3 alive nodes skip SCH selection —
        the non-PCH node acts as a regular CM.
        """
        schs: dict[int, Sensor] = {}

        for cid, members in self._clusters.items():
            pch = self._pchs.get(cid)
            if pch is None:
                continue

            alive = [s for s in members if s.is_alive and s is not pch]
            if len(alive) < 2:
                # Need at least 2 non-PCH nodes (1 SCH + 1 CM minimum)
                continue

            best_s = None
            best_chance = -1.0

            for s in alive:
                dist_pch = s.distance_to(pch)
                chance = self._fuzzy_sch_chance(s.e_res, dist_pch)

                if (chance > best_chance or
                        (chance == best_chance and best_s is not None
                         and s.e_res > best_s.e_res)):
                    best_chance = chance
                    best_s = s

            if best_s is not None:
                best_s.is_ch = True
                schs[cid] = best_s

        self._schs = schs
        return schs

    # ------------------------------------------------------------------ #
    #  K-Means                                                             #
    # ------------------------------------------------------------------ #

    def _kmeans(self, positions: np.ndarray, k: int,
                max_iter: int = 100,
                init_centroids: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Standard K-Means clustering.

        Returns (assignments, centroids) where assignments is (N,) int
        and centroids is (k, 2).
        """
        n = len(positions)
        rng = np.random.default_rng()

        if init_centroids is not None:
            centroids = init_centroids.copy()
        else:
            indices = rng.choice(n, size=k, replace=False)
            centroids = positions[indices].copy()

        assignments = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign each point to nearest centroid
            dists = np.linalg.norm(positions[:, None, :] - centroids[None, :, :],
                                   axis=2)  # (n, k)
            new_assignments = np.argmin(dists, axis=1)

            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Update centroids
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    centroids[j] = positions[mask].mean(axis=0)

        return assignments, centroids

    # ------------------------------------------------------------------ #
    #  Gap statistic                                                       #
    # ------------------------------------------------------------------ #

    def _compute_wk(self, positions: np.ndarray, assignments: np.ndarray,
                    k: int) -> float:
        """Within-cluster dispersion W_k."""
        wk = 0.0
        for j in range(k):
            mask = assignments == j
            if mask.sum() <= 1:
                continue
            cluster_pts = positions[mask]
            centroid = cluster_pts.mean(axis=0)
            wk += np.sum((cluster_pts - centroid) ** 2)
        return wk

    def _gap_statistic(self, positions: np.ndarray,
                       k_range: range) -> int:
        """Determine optimal number of clusters via Gap statistic."""
        rng = np.random.default_rng(seed=0)
        n = len(positions)

        # Bounding box for reference datasets
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)

        gaps = []
        sks = []

        for k in k_range:
            # Actual data
            assignments, _ = self._kmeans(positions, k, self._km_max_iter)
            log_wk = math.log(max(self._compute_wk(positions, assignments, k),
                                  1e-10))

            # Reference datasets
            ref_log_wks = []
            for _ in range(self._gap_n_ref):
                ref_data = rng.uniform(mins, maxs, size=(n, 2))
                ref_assign, _ = self._kmeans(ref_data, k, self._km_max_iter)
                ref_wk = self._compute_wk(ref_data, ref_assign, k)
                ref_log_wks.append(math.log(max(ref_wk, 1e-10)))

            ref_mean = np.mean(ref_log_wks)
            ref_std = np.std(ref_log_wks)
            sk = ref_std * math.sqrt(1 + 1 / self._gap_n_ref)

            gaps.append(ref_mean - log_wk)
            sks.append(sk)

        # Standard Gap criterion: smallest k where Gap(k) >= Gap(k+1) - s_{k+1}
        k_list = list(k_range)
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - sks[i + 1]:
                chosen = k_list[i]
                print(f'Gap statistic: optimal k={chosen} '
                      f'(gap={gaps[i]:.3f})')
                return chosen

        # Fallback: return k with max gap
        best_idx = int(np.argmax(gaps))
        chosen = k_list[best_idx]
        print(f'Gap statistic: fallback k={chosen} '
              f'(gap={gaps[best_idx]:.3f})')
        return chosen

    # ------------------------------------------------------------------ #
    #  PSO + K-Means hybrid                                                #
    # ------------------------------------------------------------------ #

    def _pso_kmeans(self, positions: np.ndarray,
                    k: int) -> tuple[np.ndarray, np.ndarray]:
        """Hybrid PSO + K-Means clustering (Eq. 6-7, 10)."""
        n = len(positions)
        dim = k * 2  # each particle = k centroids flattened to 2k

        rng = np.random.default_rng(seed=1)

        # Bounding box
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)

        # Initialize one particle from K-Means, rest random
        km_assignments, km_centroids = self._kmeans(positions, k,
                                                     self._km_max_iter)

        particles = np.zeros((self._pso_particles, dim))
        particles[0] = km_centroids.flatten()
        for i in range(1, self._pso_particles):
            centroids_rand = rng.uniform(mins, maxs,
                                         size=(k, 2))
            particles[i] = centroids_rand.flatten()

        velocities = np.zeros((self._pso_particles, dim))

        def _fitness(particle: np.ndarray) -> float:
            """Quantization error J (Eq. 10)."""
            centroids = particle.reshape(k, 2)
            # K-Means refinement with these centroids as init
            assign, refined = self._kmeans(positions, k,
                                           max_iter=10,
                                           init_centroids=centroids)
            total = 0.0
            for j in range(k):
                mask = assign == j
                count = mask.sum()
                if count == 0:
                    continue
                dists = np.linalg.norm(positions[mask] - refined[j], axis=1)
                total += dists.mean()
            return total / k

        # Evaluate initial fitness
        fitness = np.array([_fitness(p) for p in particles])
        pbest = particles.copy()
        pbest_fit = fitness.copy()
        gbest_idx = int(np.argmin(fitness))
        gbest = particles[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]

        # Bounds for clamping (tile for k centroids)
        lower = np.tile(mins, k)
        upper = np.tile(maxs, k)

        # PSO iterations
        for _ in range(self._pso_max_iter):
            r1 = rng.random((self._pso_particles, dim))
            r2 = rng.random((self._pso_particles, dim))

            velocities = (self._pso_w * velocities
                          + self._pso_c1 * r1 * (pbest - particles)
                          + self._pso_c2 * r2 * (gbest - particles))
            particles = particles + velocities

            # Clamp positions to data bounding box
            particles = np.clip(particles, lower, upper)

            # Evaluate
            fitness = np.array([_fitness(p) for p in particles])

            # Update personal bests
            improved = fitness < pbest_fit
            pbest[improved] = particles[improved]
            pbest_fit[improved] = fitness[improved]

            # Update global best
            best_this = int(np.argmin(fitness))
            if fitness[best_this] < gbest_fit:
                gbest = particles[best_this].copy()
                gbest_fit = fitness[best_this]

        # Final K-Means refinement on global best
        final_centroids = gbest.reshape(k, 2)
        final_assign, final_centroids = self._kmeans(
            positions, k, self._km_max_iter,
            init_centroids=final_centroids)

        print(f'PSO+K-Means: k={k}, J={gbest_fit:.3f}')
        return final_assign, final_centroids

    # ------------------------------------------------------------------ #
    #  Single round                                                        #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        """Execute one simulation round. Returns False when network is dead."""
        net = self.net

        # Reset
        net.reset_round()
        for s in net.sensors:
            s.is_ch = False
            s.ch_neighbors = []
            if s.is_alive:
                s.power = net.p_max / 4
                s.rc = net.calc_comm_range(s.power)

        # Neighbour discovery (needed for range-checked cluster formation)
        net.discover_neighbors()

        # PCH selection (fuzzy logic)
        self._select_pch()

        # SCH selection (fuzzy logic, requires PCH)
        self._select_sch()

        print(f'FL-LEACH-PSO Round {self.t}: '
              f'PCHs={len(self._pchs)}, SCHs={len(self._schs)}, '
              f'Dead={self.dead_nodes}/{net.num_nodes}')

        # Cluster formation with range checks
        self._cluster_formation()
        self._filter_neighbours()
        self._connect_unaffiliated()
        self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # Energy deduction
        self._steady_state()

        if self.dead_nodes >= net.num_nodes:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Cluster formation (range-checked, same pattern as LEACH/GTFR)       #
    # ------------------------------------------------------------------ #

    def _cluster_formation(self):
        """CHs (PCH + SCH) advertise at max power; CMs within range join nearest CH."""
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
        """Unaffiliated nodes raise power to reach a cluster."""
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
    #  Energy deduction                                                    #
    # ------------------------------------------------------------------ #

    def _steady_state(self) -> None:
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
