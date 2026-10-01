# Cluster-Aware Routing Tree & Energy Model — Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Author:** brainstormed with Claude

## Summary

Today every algorithm — clustering and topology-control alike — routes its
per-round energy over `NetworkModel.build_routing_tree()`, a **shortest-hop
path over the physical proximity graph**. That tree ignores cluster roles: a
cluster *member* near the BS connects straight to the sink in one hop, bypassing
its cluster head. Because `compute_maintenance_costs()` derives each node's
energy directly from this tree (`tx_dist`, `num_descendants`), the energy model
of the clustering protocols does **not** reflect how they actually operate
(CM → multi-hop intra-cluster relay → CH → multi-hop CH backbone → BS), and the
`avg_hop` metric reads artificially low.

This design adds a **cluster-aware routing tree** used by the 7 clustering
protocols for *both* energy and the hop/PDR metric, so the simulation charges
energy along the real cluster data path. The 4 topology-control games
(DIA, MIA, TCLE, EFTCG-1/2) are unchanged — their clusterless shortest-path
model is correct as-is.

```
  CLUSTERING algos (GT2, LEACH, GTFR, FL-LEACH-PSO, SCA-Levy, FC-CRA, EE-TCM)
      build_cluster_routing_tree(net)  ──► compute_cluster_maintenance_costs()  (energy)
                     │                  └─► metrics.record_round()              (delivered / avg_hop)
  TOPOLOGY games (DIA, MIA, TCLE, EFTCG-1/2)   [UNCHANGED]
      build_routing_tree(net)          ──► compute_maintenance_costs()
                                        └─► metrics.record_round()
```

## Decisions (locked during brainstorming)

| Question | Decision |
|----------|----------|
| CH→BS routing | **Uniform multi-hop for all 7 clustering algos** over the CH-only graph (overrides LEACH/GT2's native single-hop). |
| Intermediate-CH relay energy | **Forward-only on the backbone (same rule as CM relay):** a CH aggregates only its **own** cluster's members into one packet (`e_agg`); it does **not** re-fuse downstream CH packets. It forwards its own aggregated packet plus every downstream CH's aggregated packet — `(1+nd_ch)` CH TX + `nd_ch` RX of aggregated packets, where `nd_ch` = backbone CH-descendants below it. |
| Cluster with no path to a gateway CH | **Energy still charged** along the full chain; the component's closest-to-BS ("terminal") CH makes a **capped best-effort BS transmission** (at `min(dist_to_bs, range(p_max))`) that fails. The cluster is marked **not delivered**. |
| CM routing | **Multi-hop within the cluster.** Members route to their CH (`ch_belong`) over a per-cluster reverse-BFS tree of **same-cluster member→member** physical links — no cross-cluster relay, no CM shortcut to the BS. A member directly in CH range stays one hop; a member out of CH range relays through same-cluster members. |
| Intermediate-CM relay energy | **Forward-only (no fusion):** the CH is the sole aggregator. A relay member forwards every descendant packet plus its own (`(1+nd)` TX + `nd` RX, `nd` = member-subtree size), so every member's raw packet arrives at the CH individually. |
| CM with no path to its CH | **Energy still charged:** the member makes a real transmission **at its own current power** — charged `calc_tx_cost(rc, 'CM')` — that fails to arrive. Marked **not delivered**; excluded from `avg_hop`. (Orphans — `ch_belong` None/dead — keep the single-node-CH treatment in §B.5.) |
| Tree sharing (metric vs energy) | **Build once per round, pass to both** (recommended; see §D). |
| avg_hop on no-delivery | **Undefined (excluded)**, not logged as 0 (folds in the earlier conflation fix; see §E). |
| Topology-control games | **Unchanged.** |

## Affected components

| File | Change |
|------|--------|
| `model.py` | Add `build_cluster_routing_tree(net)` and `compute_cluster_maintenance_costs(tree)`. Add a `delivers` field to the dict schema (and set `delivers=True` in `build_routing_tree` for uniformity). |
| `algos/__init__.py` | `BaseAlgorithm`: store `self._routing_tree`; `run()` passes it to `record_round`. |
| `algos/{gt2,leach,gtfr,fl_leach_pso,sca_levy,fc_cra,ee_tcm}.py` | `_maintenance` builds the cluster tree, stashes it, deducts `costs[s.id]` for **every** alive node (removes the per-algo `is_ch` cost block). |
| `metrics.py` | `record_round(..., routing_tree=None)`: use the passed tree; derive `delivered`/`avg_hop` from `delivers`; record `avg_hop = None` when nothing delivers; `finalize`/`time_series` skip `None`. |
| `paper_tables.py` | Average `avg_hop` only over delivering seed-rounds; `--` when none. |
| `benchmark.py` / results | Delete + re-run the clustering-algo runs (see §F). |

---

## B. `build_cluster_routing_tree(net)`

Returns `dict[int, dict]` with the **same schema** as `build_routing_tree` plus
one field:

```
parent_id: int | None        # next hop toward BS (None = transmits to BS)
tx_dist: float               # distance to that next hop (or to BS)
depth: int                   # hop count to BS along the cluster path
num_descendants: int         # subtree size (for relay/aggregation accounting)
delivers: bool               # True iff this node's path actually reaches the BS
```

Total hops to BS for a node = `depth + 1` (the `+1` is the terminal gateway-CH→BS
hop), identical to the topology convention, so `avg_hop` is computed the same way.

Construction:

1. **Gateway CHs** — CHs with `dist_to_bs ≤ rc`. `parent_id = None`,
   `tx_dist = dist_to_bs`, `depth = 0`, `delivers = True`.
2. **Reachable CH backbone** — over the CH-only graph (`ch_neighbors`), reverse-BFS
   from the gateway CHs. Each reached CH gets `parent_id` = the neighbor it was
   reached from (next hop toward a gateway), `depth` = CH-hops, `delivers = True`.
3. **Stranded CH components** (no gateway in the component) — pick the
   **terminal CH** = the component's CH with the smallest `dist_to_bs`. Reverse-BFS
   from the terminal so every CH gets a next hop toward it. The terminal's
   `parent_id = None`, `tx_dist = min(dist_to_bs, calc_comm_range(p_max))`
   (capped failed attempt), and **all** nodes in the component get
   `delivers = False`.
4. **Member layer (multi-hop CM → CH).** For each CH already in the tree
   (gateway or backbone), run a **reverse-BFS rooted at the CH** over
   **same-cluster** member→member physical edges (`self.edges`, restricted to
   members whose `ch_belong` is this CH). A member `m` joins with `parent = x`
   when the directed edge `m→x` exists and `x` is already in the tree (`x` = the
   CH or another same-cluster member already reached). Each reached member gets
   `tx_dist = dist(m, parent)`, `depth = parent.depth + 1`,
   `delivers = CH.delivers`. A member directly in CH range is a one-hop direct
   child (unchanged from the old single-hop behaviour); a member out of CH range
   relays through same-cluster members.
5. **No-path members & orphans.** A member never reached in step 4:
   - **`ch_belong` alive but unreachable** (no direct or relayed path to its CH):
     `parent_id = None`, `tx_dist = rc`, `delivers = False`, excluded from
     `avg_hop`. It still spends a real TX at its own power
     (`calc_tx_cost(rc, 'CM')` in §C) that fails to arrive.
   - **Orphan** (`ch_belong is None` or the CH died this round): treated as a
     single-node CH for the round — gateway if within `rc` of BS (delivers),
     else stranded terminal (`tx_dist = min(dist_to_bs, range(p_max))`,
     `delivers = False`).
6. `num_descendants` (member-subtree size, used for forward-only relay cost) is
   computed over the finished tree. Each CH's energy uses two counts derived in
   §C from the finished tree: `M_ch` = members of its cluster that were reached
   (every one delivers a raw packet to it), and `nd_ch` = the number of CH
   descendants on the backbone below it (the backbone is forward-only, so every
   downstream CH's aggregated packet is relayed through it).

Notes / edge cases:
- The CH graph is taken from `ch_neighbors`; if asymmetric (CHs at different
  power), treat an edge as usable when present in either direction. The member
  graph uses the directed `self.edges` as-is (forward `m→x` = `m` can transmit to
  `x`), matching the directed-connectivity convention.
- Dead CHs: a CM whose `ch_belong` died this round is re-pointed to nothing →
  treated as an orphan (step 5).
- Member relays are confined to the member's own cluster; a member never relays
  through another cluster's member, preserving the cluster abstraction.

## C. `compute_cluster_maintenance_costs(net, tree)`

Per node, **all charged regardless of `delivers`** (a stranded cluster still
spends transmission energy). Sensing/processing (`c_sense`, `c_process`) are as
in the current `compute_maintenance_costs`. Packet sizes follow `calc_tx_cost`:
a **member data packet is `m_s` (`m_pkt_s`)**, a **CH aggregated packet is `m_l`
(`m_pkt_l`)**, and a node's RX cost uses the size of the packet it receives.

- **Member (forward-only — leaf, relay, and no-path share one formula):**
  `c_sense + c_process`
  `+ (1 + nd) · calc_tx_cost(tx_dist, 'CM')`   (forward own packet + each of `nd`
                                                descendant packets; no fusion)
  `+ nd · (m_s · e_elec)`                       (RX each descendant member packet)

  where `nd` = member-subtree size below it. A **leaf** member has `nd = 0`
  (one TX to its parent). A **no-path** member also has `nd = 0` with
  `tx_dist = rc`, so it pays exactly `calc_tx_cost(rc, 'CM')` for the failed
  attempt.
- **CH (forward-only backbone, aggregates only its own cluster):**
  `c_sense + c_process`
  `+ M_ch · (m_s · e_elec)`             (RX every member raw packet of its cluster)
  `+ nd_ch · (m_l · e_elec)`            (RX each downstream CH's aggregated packet)
  `+ m_l · e_agg`                       (fuse its OWN cluster's members → one packet)
  `+ (1 + nd_ch) · calc_tx_cost(tx_dist, 'CH')`
                                        (forward its own packet + every downstream
                                        CH's aggregated packet; capped BS attempt
                                        for a stranded terminal)

  where `M_ch` = reached members of its cluster and `nd_ch` = the number of CH
  descendants on the backbone below it (CHs that forward through it). The CH
  aggregates only its own cluster's data; the backbone is **forward-only** (no
  re-fusion), so a downstream CH's packet is RX'd and re-TX'd at every hop —
  exactly the CM relay rule, one level up. Counted from the finished tree:
  `nd_ch` is the CH-only subtree size (each CH `c` adds `1 + nd_ch[c]` to
  `nd_ch[parent(c)]`); a reached member `m` (parent not `None`) contributes to
  `M_ch[m.ch_belong]`.

This **replaces** the `is_ch` cost block currently duplicated in each clustering
algo's `_maintenance`; each `_maintenance` becomes: build tree → compute costs →
`s.e_res -= costs[s.id]` for every alive node → death tracking. The shared
`NetworkModel` energy constants are untouched (the project's energy-model
invariant holds: only the routing topology fed into the formulas changes).

Consequence to expect: because every alive clustered node now transmits every
round (today, unrouted nodes paid sensing only), **per-round energy for the
clustering algos rises**, shortening their lifetimes versus the current numbers.
Forward-only relaying compounds this — members on an intra-cluster relay path pay
for every descendant packet they carry, so members near a CH drain faster and
`avg_hop` lengthens. That is the intended, more-faithful model.

## D. Plumbing — metric and energy share one tree

The hop/PDR metric must be computed from the **same** tree that energy was
charged on, or the reported hops would contradict the charged energy.

**Chosen approach — build once, pass through:**
- `BaseAlgorithm` gains `self._routing_tree = None`.
- Each algo's `_maintenance` sets `self._routing_tree = tree` after building it
  (clustering algos: the cluster tree; topology games: their existing physical
  tree).
- `run()` calls `record_round(self.net, self.t, extras, routing_tree=self._routing_tree)`.
- `record_round(..., routing_tree=None)` uses the passed tree; if `None`
  (defensive fallback), it builds `build_routing_tree` as today.

This guarantees metric == energy and removes the current redundant second
`build_routing_tree()` call inside `record_round`.

*Alternative considered (not chosen):* family-dispatch inside `record_round`
(rebuild a cluster tree for the clustering family, physical for topology). Lower
touch but rebuilds the tree a second time each round and risks metric/energy
divergence if the two build calls ever see different state.

## E. `avg_hop` aggregation fix (folds in the earlier conflation finding)

Both trees now carry `delivers`. In `record_round`:
- `delivered = count(nodes with delivers == True)`.
- `avg_hop = mean(depth + 1 over delivering nodes)`, or **`None`** when nothing
  delivers (no longer `0.0` — that conflated "no path" with "0 hops").
- `finalize()` computes `mean_avg_hop` over non-`None` rounds only;
  `time_series` stores `None` for no-delivery rounds.
- `paper_tables.py` averages a milestone cell only over seeds whose value is not
  `None`; renders `--` when every contributing seed had no delivery.

`avg_tx_power`, `total_energy`, `energy_std`, `alive` are unaffected.

## F. Re-run & validation plan

1. **Unit tests first** (`tests/`), before any sweep:
   - `build_cluster_routing_tree`: hand-built 5–6 node fixtures covering
     (a) a gateway CH with members, (b) a 3-CH backbone to one gateway
     (depths 0/1/2), (c) a stranded component (no gateway → all `delivers=False`,
     terminal capped), (d) an orphan CM.
   - `compute_cluster_maintenance_costs`: assert CM = sense+proc+1 TX; CH =
     sense+proc + RX·children + agg + 1 TX; numbers match a hand calc.
   - Metric: `avg_hop = None` when no node delivers; equals the hand hop count
     otherwise.
2. **Smoke run** each clustering algo for ~3 rounds (`max_rounds=3`) to confirm no
   crashes and energy strictly decreases.
3. **Re-sweep the clustering algos only.** Delete
   `results/runs/{GT2,LEACH,GTFR,FL-LEACH-PSO,SCA-LEVY,FC-CRA}_*.json` and re-run
   `benchmark.py` (resumable → the 5 topology-game algos are skipped, their JSON
   kept). EE-TCM is in the 7 but not in `benchmark.ALGOS`, so it is implemented
   and unit-tested but not in the sweep. ≈ 6 algos × 6 scenarios × 10 seeds =
   **360 runs**.
4. **Regenerate outputs:** `plot_benchmark.py` (figures + `summary_by_scenario.csv`),
   then `paper_tables.py`; refill the paper's hop / tx-power / epp / lifetime
   tables and figures.
5. **Paper note:** add one sentence to the methodology that clustering protocols
   are charged along their CM→CH→backbone→BS data path (topology games over the
   shared physical graph), so the comparison basis is each protocol's real
   operation.

## Risks

- **All clustering-algo results change** (FND/HND/LND, energy curves, PDR, hops).
  Expected and intended; the paper text/tables must be regenerated together.
- **CH-backbone connectivity depends on CH power** (`ch_neighbors` is range-based).
  If CH power is low, components fragment and more clusters strand — captured
  honestly by `delivers=False` + the energy-still-charged rule.
- **Per-algo `_maintenance` divergence:** FC-CRA and SCA-Lévy already have bespoke
  multi-hop CH handling; the shared model supersedes it. Verify their cluster
  structures (`ch_belong`, `ch_neighbors`) are populated before `_maintenance`
  runs in each.

## Out of scope

- Changing energy constants or the Friis/energy formulas.
- Topology-control games' routing/energy.
- Re-running or altering the deployment scenarios.
