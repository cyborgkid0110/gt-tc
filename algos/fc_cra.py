"""
FC-CRA: Clustering and Routing Algorithm for Fast Changes of Large-Scale WSN.

Per round:
  1. Build an adaptive per-node cluster radius
        R(i) = (1 + β_i) · α_i · R_0
     where β_i blends an energy factor f_E and a distance-to-BS factor f_D
     weighted by an energy-dispersion coefficient D_E; R_0 shrinks as the network
     ages; α_i compensates local density.
  2. Iterative CH election: repeatedly pick the node with the largest CHCC
     (∂_i = |h_i|, the count of alive in-radius neighbours) whose CHCC is ≥ the
     average threshold ∂_th. Mark its neighbours as clustered. Remaining nodes
     join their nearest CH.
  3. Materialise CHs on the NetworkModel, run the shared cluster-formation /
     neighbour filtering / unaffiliated-node repair flow, then a layered-batch
     intra-cluster maintenance and a multi-hop CH→CH→BS forwarding pass.

  Clusters persist across rounds and are only rebuilt when a CH's residual
  energy falls below `z_realloc_pct` of its level at the last rebuild, or when
  a CH has died.

Notes (see PLAN.md):
  - Uses the project's Friis energy model via `NetworkModel.calc_tx_cost` /
    `calc_node_cost`, not the paper's two-regime first-order radio model.
  - `d_max` is expressed as `d_max_factor · calc_comm_range(p_max)` to replace
    the paper's `1.2·d_0` (d_0 is radio-model specific and unavailable here).
  - Transmission power is clamped to `[p_min, p_max]` after translating R(i)
    back to a power via the inverse Friis relation.
  - The paper's IV-C Path-Energy-Function routing and IV-D ICCNS Dijkstra
    routing are implemented as reference helpers (`_intra_pef_route` and
    `_inter_iccns_route`) but are **not** wired into the main maintenance
    loop, per PLAN.md.

Reference: Fan, B., & Xin, Y. (2024). A Clustering and Routing Algorithm for
           Fast Changes of Large-Scale WSN in IoT. IEEE Internet of Things
           Journal, 11(3), 5036–5049.
"""

import math
import heapq
import yaml
import numpy as np

from algos import BaseAlgorithm
from model import NetworkModel, Sensor
from plot import directional_wsn_plot
import graph


class FCCRA(BaseAlgorithm):
    """FC-CRA clustering + multi-hop routing algorithm."""

    def __init__(self, net: NetworkModel, config_path: str = 'config/fc_cra.yaml', **kwargs):
        super().__init__(net, config_path, **kwargs)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self._P: float = float(cfg['P'])
        self._d_max_factor: float = float(cfg['d_max_factor'])
        self._z_pct: float = float(cfg['z_realloc_pct'])
        self._chcc_avg_mode: str = str(cfg.get('chcc_avg_mode', 'mean')).lower()

        self._d_max: float = self._d_max_factor * net.calc_comm_range(net.p_max)
        # Deployment half-diagonal for R_m (nodes in main.py are placed in a
        # 2·AREA × 2·AREA centred square → half-diagonal = AREA · √2).
        self._R_m: float = float(net.area) * math.sqrt(2.0)

        # Cluster persistence state
        self._cluster_stable: bool = False
        self._chs: list[Sensor] = []
        self._ch_of: dict[int, Sensor] = {}       # sensor_id → CH
        self._ch_e_ref: dict[int, float] = {}     # CH.id → energy snapshot

    # ------------------------------------------------------------------ #
    #  Round                                                               #
    # ------------------------------------------------------------------ #

    def _run_round(self) -> bool:
        net = self.net
        net.reset_round()

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

        # ---- Phase 1: (re)build clusters if needed ------------------
        if not self._cluster_stable or self._needs_realloc():
            self._rebuild_clusters(alive)
            self._cluster_stable = True

        # ---- Phase 2: materialise cluster structure -----------------
        self._apply_clusters_to_net()

        num_ch = sum(1 for s in net.sensors if s.is_ch)
        print(f'FC-CRA Round {self.t}: CHs={num_ch}, '
              f'Dead={self.dead_nodes}/{net.num_nodes}')
        self._filter_neighbours()
        self._connect_unaffiliated()
        self._cleanup_cross_cluster_edges()

        if self.t % self.plot_period == 0:
            directional_wsn_plot(net.to_network_dict(), net.to_node_dict())

        # ---- Phase 3: multi-hop CH→BS routes ------------------------
        ch_routes = self._build_ch_to_bs_routes()

        # ---- Phase 4: maintenance -----------------------------------
        self._maintenance(ch_routes)

        if self.dead_nodes >= net.num_nodes:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Phase 1: clustering                                                 #
    # ------------------------------------------------------------------ #

    def _rebuild_clusters(self, alive: list[Sensor]) -> None:
        """Compute adaptive R(i) for every alive node, then iteratively elect
        CHs by greedy CHCC until no candidate beats the threshold."""
        n_alive = len(alive)
        e_res = np.array([s.e_res for s in alive], dtype=float)

        # --- E_th = median residual energy --------------------------
        e_th = float(np.median(e_res))
        e_max = float(e_res.max())
        e_mean = float(e_res.mean())
        e_std = float(e_res.std())
        D_E = min(1.0, e_std / e_mean) if e_mean > 0 else 0.0

        # --- f_E: zero below median, scaled to [0,1] above ----------
        if e_max > e_th:
            f_E = np.clip((e_res - e_th) / (e_max - e_th), 0.0, 1.0)
        else:
            f_E = np.zeros_like(e_res)
        f_E_max = float(f_E.max()) if f_E.max() > 0 else 1.0

        # --- f_D: zero within d_max, scaled to [0,1] outside --------
        d_bs = np.array([math.hypot(s.x, s.y) for s in alive], dtype=float)
        denom = max(self._R_m - self._d_max, 1e-9)
        f_D = np.clip((d_bs - self._d_max) / denom, 0.0, 1.0)

        # --- β_i ----------------------------------------------------
        beta = D_E * (f_E / f_E_max) + (1.0 - D_E) * f_D

        # --- R_0: shrinks as network ages ---------------------------
        e0 = alive[0].e0
        R_0 = math.sqrt((self._R_m ** 2) * e_th
                        / max(1, n_alive) / max(self._P * e0, 1e-12))

        # --- α_i: local density correction --------------------------
        # N_0(i) = count of alive nodes within R_0 of i (including i).
        positions = np.array([[s.x, s.y] for s in alive], dtype=float)
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        within_R0 = d2 <= R_0 ** 2
        N_0 = within_R0.sum(axis=1).astype(float)  # includes self, never zero
        alpha = np.sqrt(1.0 / np.maximum(self._P * N_0, 1e-12))

        # --- R(i) --------------------------------------------------
        R_i = (1.0 + beta) * alpha * R_0

        # --- CHCC: ∂_i = |{ j≠i, j unmarked : d(i,j) ≤ R(i) }| ----
        # Recomputed after every CH pick over still-unmarked nodes (paper
        # §IV-B: "marked nodes are removed from subsequent iterations").
        d = np.sqrt(d2)
        within_R = (d <= R_i[:, None]) & (np.arange(n_alive)[None, :]
                                           != np.arange(n_alive)[:, None])

        # --- iterative CH selection --------------------------------
        marked = np.zeros(n_alive, dtype=bool)
        chs: list[Sensor] = []
        ch_of: dict[int, Sensor] = {}

        while True:
            unmarked = ~marked
            if not unmarked.any():
                break
            # CHCC over the current unmarked set; self-row is masked by the
            # inequality in `within_R`.
            cchc = (within_R & unmarked[None, :]).sum(axis=1).astype(int)
            # threshold recomputed over still-unmarked candidates
            cchc_unmarked = cchc[unmarked]
            if self._chcc_avg_mode == 'median':
                cchc_th = float(np.median(cchc_unmarked))
            else:
                cchc_th = float(cchc_unmarked.mean())

            eligible = unmarked & (cchc >= cchc_th)
            if not eligible.any():
                break
            # pick highest CHCC; ties → smaller sensor id for determinism
            cand_idx = np.where(eligible)[0]
            scores = cchc[cand_idx]
            best = cand_idx[np.lexsort((np.array(
                [alive[i].id for i in cand_idx]), -scores))[0]]

            ch = alive[best]
            chs.append(ch)
            marked[best] = True
            ch_of[ch.id] = ch

            # mark every unmarked node in h_i as clustered members of this CH
            members = np.where(within_R[best] & ~marked)[0]
            for mi in members:
                m = alive[int(mi)]
                marked[int(mi)] = True
                ch_of[m.id] = ch

        # --- leftover unclustered nodes → nearest CH ---------------
        if chs:
            ch_xy = np.array([[c.x, c.y] for c in chs], dtype=float)
            for i in range(n_alive):
                s = alive[i]
                if s.id in ch_of:
                    continue
                dd = ((ch_xy - np.array([s.x, s.y])) ** 2).sum(axis=1)
                ch_of[s.id] = chs[int(np.argmin(dd))]
        else:
            # degenerate: no candidate ever cleared the threshold. Fall back to
            # picking the node with the highest initial CHCC across all alive
            # nodes and assigning every other alive node to it.
            cchc0 = within_R.sum(axis=1).astype(int)
            best = int(np.argmax(cchc0))
            ch = alive[best]
            chs.append(ch)
            ch_of = {s.id: ch for s in alive}

        # --- commit state -----------------------------------------
        self._chs = chs
        self._ch_of = ch_of
        self._ch_e_ref = {c.id: c.e_res for c in chs}

        print(f'  rebuild: N_alive={n_alive} CHs={len(chs)} '
              f'R_0={R_0:.2f} D_E={D_E:.3f}')

    def _needs_realloc(self) -> bool:
        """Trigger rebuild if any CH died or fell below z_pct of its snapshot."""
        if not self._chs:
            return True
        for c in self._chs:
            if not c.is_alive:
                return True
            e_ref = self._ch_e_ref.get(c.id, c.e_res)
            if e_ref > 0 and c.e_res / e_ref <= self._z_pct:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Phase 2: apply clusters to the NetworkModel                         #
    # ------------------------------------------------------------------ #

    def _apply_clusters_to_net(self) -> None:
        """Set is_ch/ch_belong, pick per-CH power to reach its farthest member
        (bounded by [p_min, p_max]), connect CHs↔members, and register
        CH↔CH edges for the multi-hop backbone."""
        net = self.net

        # Drop dead CHs from the persistent set before materialising.
        self._chs = [c for c in self._chs if c.is_alive]
        self._ch_of = {sid: ch for sid, ch in self._ch_of.items()
                       if ch.is_alive}

        for ch in self._chs:
            ch.is_ch = True

        # Power selection per CH: reach the farthest live member.
        for ch in self._chs:
            members = [net.sensors[sid] for sid, c in self._ch_of.items()
                       if c is ch and sid != ch.id
                       and net.sensors[sid].is_alive]
            if members:
                d_max = max(ch.distance_to(m) for m in members)
            else:
                d_max = 0.0
            power = self._power_for_range(d_max)
            power = max(net.p_min, min(net.p_max, power))
            ch.power = power
            ch.rc = net.calc_comm_range(ch.power)

        # Connect CH → each live member (within rc after power set above).
        for sid, ch in self._ch_of.items():
            if sid == ch.id:
                continue
            m = net.sensors[sid]
            if not m.is_alive or m.is_ch:
                continue
            if ch.distance_to(m) <= ch.rc:
                m.ch_belong = ch
                net.connect(ch, m)

        # CH ↔ CH adjacency (for multi-hop CH→BS backbone).
        for i, c1 in enumerate(self._chs):
            for c2 in self._chs[i + 1:]:
                d = c1.distance_to(c2)
                if d <= c1.rc:
                    c1.ch_neighbors.append(c2)
                    net.edges[c1.id, c2.id] = 1
                if d <= c2.rc:
                    c2.ch_neighbors.append(c1)
                    net.edges[c2.id, c1.id] = 1

    def _power_for_range(self, d: float) -> float:
        """Inverse of NetworkModel.calc_comm_range:
           P_t = d² · p_th · 16π² / λ².
        Returns p_min when d = 0 so a CH with no members still transmits."""
        net = self.net
        if d <= 0:
            return net.p_min
        return (d ** net.gamma * net.p_th
                * (4 * math.pi / net.wave) ** net.gamma
                / (net.g_ant * net.eta))

    # ------------------------------------------------------------------ #
    #  Shared cluster-repair helpers (copy pattern from LEACH/SCA-Lévy)   #
    # ------------------------------------------------------------------ #

    def _filter_neighbours(self):
        net = self.net
        for s in net.sensors:
            if not s.is_alive:
                continue
            if s.is_ch:
                for nb in s.neighbors[:]:
                    if nb.is_ch:
                        # Keep CH↔CH adjacency in ch_neighbors for the backbone,
                        # but remove the CM-style edge so intra-cluster routines
                        # don't treat another CH as a cluster member. Use
                        # `net.disconnect` so `net.edges` stays consistent with
                        # `s.neighbors` (directed convention).
                        net.disconnect(s, nb)
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
    #  Phase 3: multi-hop CH→BS backbone (monotone progress toward BS)    #
    # ------------------------------------------------------------------ #

    def _build_ch_to_bs_routes(self) -> dict[int, list[Sensor]]:
        """For each alive CH, return the chain of CH hops used to reach the BS.
        Greedy next-hop = the ch_neighbor strictly closer to the BS than self.
        """
        routes: dict[int, list[Sensor]] = {}
        chs = [c for c in self._chs if c.is_alive]
        if not chs:
            return routes

        def d_bs(s: Sensor) -> float:
            return math.hypot(s.x, s.y)

        adj: dict[int, list[Sensor]] = {c.id: [] for c in chs}
        for c in chs:
            dc = d_bs(c)
            for nb in c.ch_neighbors:
                if nb.is_alive and nb.is_ch and d_bs(nb) < dc:
                    adj[c.id].append(nb)

        for c in chs:
            chain: list[Sensor] = []
            cur = c
            visited = {cur.id}
            while True:
                hops = adj[cur.id]
                if not hops:
                    break
                nxt = min(hops, key=d_bs)
                if nxt.id in visited:
                    break
                chain.append(nxt)
                visited.add(nxt.id)
                cur = nxt
            routes[c.id] = chain
        return routes

    # ------------------------------------------------------------------ #
    #  Phase 4: maintenance                                                #
    # ------------------------------------------------------------------ #

    def _maintenance(self, ch_routes: dict[int, list[Sensor]]) -> None:
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
                    self._cluster_stable = False
            elif s.id in costs:
                s.c_cm = costs[s.id]
                s.e_res -= s.c_cm
                if s.e_res <= 0:
                    self._track_death(s)

    # ------------------------------------------------------------------ #
    #  Reference helpers (NOT on the main flow — per PLAN.md)              #
    # ------------------------------------------------------------------ #

    def _intra_pef_route(self, ch: Sensor,
                          l_bits: int | None = None) -> dict[int, int]:
        """Paper §IV-C Path-Energy-Function routing inside one cluster.

        Forward-cone FNS: j is a valid forward hop for i iff
            d(i,j)² + d(j, CH)² ≤ d(i, CH)².
        Next hop of each node = the FNS member maximising the minimum residual
        energy along the resulting path (computed recursively outward from CH).

        Returns {sensor_id: next_hop_sensor_id}.
        """
        net = self.net
        if l_bits is None:
            l_bits = net.m_pkt_s
        members = [s for s in net.sensors
                   if s.is_alive and s.ch_belong is ch and s is not ch]
        if not members:
            return {}

        d_to_ch = {m.id: m.distance_to(ch) for m in members}
        d_to_ch[ch.id] = 0.0

        # Order members by hop distance to CH (shortest first).
        members_sorted = sorted(members, key=lambda m: d_to_ch[m.id])

        # F(ch, ...) seed. Start with each member forwarding directly to CH
        # and refine by evaluating candidate FNS members.
        next_hop: dict[int, int] = {ch.id: ch.id}
        f_val: dict[int, float] = {ch.id: ch.e_res}

        for m in members_sorted:
            # Build FNS: closer-to-CH nodes already processed satisfying cone.
            best_nh = ch
            best_f = min(
                m.e_res - net.calc_tx_cost(d_to_ch[m.id], 'CM'),
                f_val[ch.id] - l_bits * net.e_elec,
            )
            for j in members_sorted:
                if j is m or j.id not in next_hop:
                    continue
                d_ij = m.distance_to(j)
                if d_ij ** 2 + d_to_ch[j.id] ** 2 > d_to_ch[m.id] ** 2:
                    continue
                lam = m.e_res - net.calc_tx_cost(d_ij, 'CM')
                cand_f = min(lam, f_val[j.id] - l_bits * net.e_elec)
                if cand_f > best_f:
                    best_f = cand_f
                    best_nh = j
            next_hop[m.id] = best_nh.id
            f_val[m.id] = best_f

        return next_hop

    def _inter_iccns_route(self) -> dict[int, list[int]]:
        """Paper §IV-D ICCNS routing. Build an intercluster communication node
        set (CHs + near-BS non-CH nodes + a relay node per isolated CH) and run
        Dijkstra on
                w_ij = E_T(L, d_ij) / E(i) + E_R(L) / E(j)
        (with E_T and E_R taken from the project energy model) to find a
        minimum-weight CH→BS path per CH.

        Returns {ch_id: [hop_id, hop_id, ..., sink_id]}. The sink id is a
        virtual node whose id is -1 (the BS at origin).
        """
        net = self.net
        BS_ID = -1
        packet = net.m_pkt_l

        # Build ICCNS node set
        iccns: list[Sensor] = [s for s in net.sensors
                                if s.is_alive and s.is_ch]
        near_bs = [s for s in net.sensors
                    if s.is_alive and not s.is_ch
                    and math.hypot(s.x, s.y) <= self._d_max]
        iccns.extend(near_bs)

        # For each CH with no other CH within self._d_max, add one relay node.
        relays: list[Sensor] = []
        ch_ids = {c.id for c in self._chs if c.is_alive}
        for c in self._chs:
            if not c.is_alive:
                continue
            nearest = min((o for o in self._chs
                           if o.is_alive and o is not c),
                          key=lambda o: c.distance_to(o), default=None)
            if nearest is None:
                continue
            if c.distance_to(nearest) <= self._d_max:
                continue
            # pick relay maximising f = E / (E_T · DT)
            best_r, best_f = None, -math.inf
            for r in net.sensors:
                if (not r.is_alive or r.id in ch_ids or r is nearest
                        or r.id in {rr.id for rr in iccns}):
                    continue
                d_cr = c.distance_to(r)
                d_rn = r.distance_to(nearest)
                if d_cr ** 2 + d_rn ** 2 > c.distance_to(nearest) ** 2:
                    continue  # not on the forward cone between c and nearest
                et = net.calc_tx_cost(d_cr, 'CH')
                dt = (d_cr + d_rn) / max(c.distance_to(nearest), 1e-9)
                if et <= 0 or dt <= 0:
                    continue
                f = r.e_res / (et * dt)
                if f > best_f:
                    best_f = f
                    best_r = r
            if best_r is not None:
                relays.append(best_r)
        iccns.extend(relays)

        # Weighted directed graph
        nodes = {s.id: s for s in iccns}
        edges: dict[int, list[tuple[int, float]]] = {sid: [] for sid in nodes}
        edges[BS_ID] = []
        for sid, s in nodes.items():
            # edge s → BS (origin)
            d = math.hypot(s.x, s.y)
            if d > 0:
                et = net.calc_tx_cost(d, 'CH')
                w = et / max(s.e_res, 1e-9)  # no receiver energy at BS
                edges[sid].append((BS_ID, w))
            for tid, t in nodes.items():
                if tid == sid:
                    continue
                d = s.distance_to(t)
                if d <= 0:
                    continue
                et = net.calc_tx_cost(d, 'CH')
                er = packet * net.e_elec
                w = et / max(s.e_res, 1e-9) + er / max(t.e_res, 1e-9)
                edges[sid].append((tid, w))

        # Dijkstra from each alive CH
        routes: dict[int, list[int]] = {}
        for c in self._chs:
            if not c.is_alive:
                continue
            dist = {BS_ID: math.inf}
            dist.update({sid: math.inf for sid in nodes})
            prev: dict[int, int] = {}
            dist[c.id] = 0.0
            heap = [(0.0, c.id)]
            while heap:
                du, u = heapq.heappop(heap)
                if du > dist[u]:
                    continue
                if u == BS_ID:
                    break
                for v, w in edges.get(u, []):
                    nd = du + w
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            if dist.get(BS_ID, math.inf) == math.inf:
                routes[c.id] = [BS_ID]  # fallback = direct hop
                continue
            # reconstruct
            path = [BS_ID]
            cur = BS_ID
            while cur in prev and cur != c.id:
                cur = prev[cur]
                path.append(cur)
            routes[c.id] = list(reversed(path))
        return routes
