# TC-GSC: Dynamic Spatial-Correlation-Aware Topology Control Using Game Theory

## Overview

- TC-GSC (Dynamic Spatial-Correlation-Aware Topology Control using Game Theory) is a topology control method for wireless sensor networks (WSNs) that combines spatial-correlation-based active node selection with a non-cooperative power-allocation game, aiming to balance residual energy across nodes, reduce redundant data transmission, and preserve network connectivity. *(Abstract; Section I)*
- The method is organized into three phases — initialization, adaptation, and maintenance — and is compared experimentally against three existing game-based algorithms: DIA, TCLE, and VGEB. *(Section I; Section V)*

## Network Model

- The network is represented as a connected undirected graph H = (N, E); a link exists when transmitting node m's power p_m ≥ w_mn, the power required to reach node n. *(Section III.A.2, Eq. 1)*
- Packet reception condition: p_m·G_mn ≥ p̃, where G_mn is the propagation factor and p̃ the signal capture threshold. *(Eq. 2)*
- Free-space propagation model: G_mn = C·d_mn^(−α), with C a constant, α the path-loss factor (2 ≤ α ≤ 6), d_mn the Euclidean distance between m and n. *(Eq. 3)*
- Link-state variable: l_mn = 1 if p_m ≥ w_mn, else 0. *(Eq. 4)*
- Bidirectional link condition: min{p_m, p_n} ≥ p^th/G_mn, giving w_mn = p^th/G_mn (with the symmetry assumption w_mn = w_nm). *(Eq. 5)*
- Neighbor set NH_n(p_m) and the joint-power-induced topology g(p) = {mn | l_mn(p_m)·l_nm(p_n) = 1; m≠n∈N} are defined; the algorithm's goal is to obtain an energy-saving connected subgraph g(p) of the maximum-power topology g_max. *(Eq. 6; Section III.A.2)*
- All links in the model are explicitly stated to be bidirectional. *(Section III.A.2)*

## Game Theory Preliminaries

- A strategic non-cooperative game is defined as Γ = ⟨B, S, u⟩: player set B = {b1,...,bn}, strategy set S as the Cartesian product of individual strategy sets S_i, and utility functions U = {u1,...,un} mapping strategy profiles to payoffs. *(Section III.A.1, Definition 1)*
- Nash equilibrium (NE) is adopted as the stability concept: a strategy profile from which no player can improve its payoff by unilateral deviation. *(Section III.A.1)*

## Spatial Correlation and Region Partitioning

- The monitoring area is divided into rectangular correlation cells of size d, exploiting the tendency of spatially close nodes to collect similar readings. *(Section III.A.3)*
- Maximum cell size: d_max = r_c·cos45°, where r_c is the node's communication radius; d can be tuned to the application and event type. *(Section III.A.3)*
- Cell index calculation for a node at (x_n, y_n) relative to sink (x_s, y_s):
  - x_c = ⌊((x_n − x_s) − d/2)/d⌋ + 1, if x_n > 0
  - x_c = ⌊((x_n − x_s) + d/2)/d⌋ − 1, if x_n < 0
  - y_c is computed by the same procedure. *(Eq. 7)*
- Nodes within the same cell alternate sensing duties rather than sensing simultaneously, which reduces redundant data transmission and saves energy. *(Section III.A.3)*

## Active Node Selection

- Exactly one active node is selected per correlation cell at any time; all other nodes in the cell remain in sleep state. *(Section III.B.1)*
- The sink node maintains status information I_n = ⟨(x_n,y_n), s_n⟩ for every node, including residual energy and fault state. *(Section III.B.1)*
- First selection round: the node nearest to the sink in each cell is chosen as active. *(Section III.B.1)*
- Subsequent rounds: the sink broadcasts a validation message; nodes that respond are marked "good" (GD) and assigned an idle-time threshold
  - T̃ = γ·exp(V_sta/V_pre) *(Eq. 8)*
  where V_sta is the battery's standard working voltage, V_pre the current voltage, and γ a tunable constant.
- Nodes that do not respond are marked "faulty" (FT). Among the GD nodes in a cell, the one with the highest residual energy is selected as the new active node; the sink then broadcasts the updated active-node assignment to all nodes. *(Section III.B.1)*

## Game Model for Power Allocation

- Player set B = {b1,...,bm} corresponds to the active node set A; each active node k adapts its transmitting power p_k(t) ∈ [0, p_k^max]. *(Section III.B.2)*
- The power vector p = (p1,...,pm) forms the strategy space S. *(Section III.B.2)*
- Utility function:
  - u_k(p_k(t), p_{−k}(t)) = f_k(p_k(t), p_{−k}(t))·(f_e(k) + f_pr(k)) − c_k(p_k(t)) *(Eq. 9)*
  - f_k(p_k(t), p_{−k}(t)) = 1 if node k can establish bidirectional-link connections to its neighbors at power p_k(t) with all intermediate link powers below p_k(t); otherwise f_k = 0. f_k is non-decreasing in p_k(t). *(Section III.B.2)*
  - f_e(k) = α·Σ(E_r(k) − Ē_r^h)² / ΣNH_k^h — the energy-balance benefit of the topology relative to the average residual energy Ē_r^h of node k's h-hop neighbors, where E_r(k) is k's residual energy and NH_k^h the number of h-hop neighbors. *(Eq. 10)*
  - f_pr(k) = β·ΣNH_k^h — the contribution to forming a connected topology, proportional to the number of h-hop neighbors. *(Eq. 11)*
  - α and β are weighting parameters chosen so that (f_e(k) + f_pr(k)) exceeds c_k(p_k(t)) when f_k = 1. *(Section III.B.2)*
- Cost function (increasing in residual energy, discouraging selfish over-transmission):
  - c_k(p_k(t)) = u·∫ g(x)dx, integrated from (E_0(k) − E_r(k)) to (E_0(k) − E_r(k) + p_k(t)·T) *(Eq. 12)*
  where E_0(k) is node k's initial energy, T the unit transmission time (with p_k(t)·T ≤ E_r(k)), u a sufficiently small scaling constant keeping c_k(p_k(t)) ∈ [0,1], and g(x) = exp(x/10) the energy-cost increment function. *(Section III.B.2)*
- The purpose of the game is to drive each active node's power selection toward a Nash equilibrium of u, yielding an energy-balanced connected topology. *(Section III.B.2)*

## TC-GSC Algorithm — Three Phases

TC-GSC consists of an initialization phase, an adaptation phase, and a maintenance phase, executed cyclically until a preset maximum control time is reached. *(Section IV)*

### Initialization Phase

- Every node i belonging to the active node set A initializes its power level to p_i^max. *(Section IV.A)*
- To discover neighbors, node i broadcasts a Neighbor Request Message (NBM) at p_i^max carrying an h-hop marker. The NBM format is: [NBM, ID_i, (x_i, y_i), p_i^max, E_0(i), E_r(i), T]. *(Section IV.A)*
- Each responding neighbor j returns a Neighbor Reply Message (NRM) with format [NRM, ID_j, (x_j, y_j), p_ij, E_0(j), E_r(j), T]; node i records j in its 1-hop neighbor list NH_i^1. *(Section IV.A)*
- Node j continues rebroadcasting the NRM to further neighbors, decrementing the hop marker at each step, until it reaches 0 — at which point node i has obtained its full h-hop neighbor set NH_i^h and its local maximum-power topology G_i^max. *(Section IV.A)*

### Adaptation Phase

- Each node's transmitting power depends on node degree, residual energy, and the topology information collected during initialization. *(Section IV.B)*
- Each node i discretizes its strategy set S into a descending sequence:
  - S_i = {p_i^max = p_i^1, p_i^2, ..., p_i^η = p_i^min} *(Eq. 13)*
- Nodes select their next power level according to:
  - p̂_i = argmax_{q_i ∈ {(p_ik, p_i(k+a))}} u_i(q_i, p_{−i}) *(Eq. 14)*
  where p_ik is node i's current power level (k = 1,...,η−1, k < k+a ≤ η), and a sufficiently small step size δ is used so that at most one link is deleted per step.
- Each node i decreases its power one level at a time if doing so yields a strictly higher payoff than its current level; otherwise it reverts to its current level. This is formalized as **Algorithm 1**, reproduced below. *(Section III.B.2; Section IV.B)*

**Algorithm 1 — Game Γ(i)** *(as given in the paper, Section IV.B)*

```
Input:  the neighbor list of node i: NH_i^1;
        the neighbor list of i-th node's h-hop neighbor: NH_i^h.
Output: the i-th node's transmitting power p_i

1:  l = 1
2:  p_i = p_i^max = p_jl^(i) ∈ p_i
3:  while p_i is not a NE do
4:      l = l + 1
5:      choose p̂^(i) = p_jl^(i) ∈ p_i
6:      for all j ∈ A(i) do
7:          if G_i^h is connected then
8:              u_j(i) = u_jl(i)
9:          else do
10:             u_j(i) = -c(i)
11:         end if
12:         p̂_i = argmax_{q_i ∈ {(p_ik, p_i(k+a))}} u_i(q_i, p_{-i})
13:     end for
14: end while
```

- Interpretation of the pseudocode: node i begins at its maximum power p_i^max (line 2). While the current power vector does not constitute a Nash equilibrium (line 3), the node evaluates an alternative power level p̂^(i) (line 5) for every node j in its active neighbor set A(i) (line 6): if the resulting local topology G_i^h remains connected, node j's utility under this candidate is retained (line 8); if connectivity would be lost, the utility is penalized to the negative of the cost term, −c(i) (line 10), discouraging that choice. The node then updates its power to the level maximizing utility over the candidate pair (p_ik, p_i(k+a)) (line 12), and the loop repeats until no further unilateral improvement is possible, i.e., until the NE is reached. *(Section IV.B)*
- Termination condition: if no node's power level is selected to further reduce transmission without dropping the connectivity of the local topology g_i, the algorithm terminates. *(Section IV.B)*
- After convergence, node i updates its local topology g_i based on neighbors' new power settings: upon receiving an NRM from neighbor j, node i checks whether a path exists between them using only intermediate nodes already in its own neighbor set; if so, NH(i) is updated with j's new power setting, otherwise j is removed from NH(i). *(Section IV.B)*

### Maintenance Phase

- The number of times the topology is adaptively reconfigured is defined as the control time of the topology, TC(t), measured in algorithm rounds r. At TC(t) = 0(r), each node i holds its initial energy E_0(i); the first power-level adaptation occurs at TC(t) = 1(r), producing the first NE, p̂(t1). *(Section IV.C)*
- Because energy consumption gradually becomes unbalanced over time, two triggered scenarios reconfigure the topology during maintenance:
  1. The i-th active node's residual energy E_r(i) falls below the threshold energy Ẽ(i).
  2. The active node's elapsed working time t_i exceeds its active time threshold T̃(i) (defined in Eq. 8). *(Section IV.C)*
- When either trigger fires, all sleeping nodes in the active node's correlation cell receive an Awake Request Message (ARM); the sleeping node with the highest residual energy is triggered to become the new active node. Each such update increments TC(t) by one. *(Section IV.C)*
- The maximum control time TC(t)_max is preset in advance, bounding the number of reconfiguration rounds. *(Section IV.C)*

### Algorithm Summary (Steps 1–7, as listed in the paper)

- **Step 1**: Partitioning of correlation regions.
- **Step 2**: Forming the active node set A; each active node i calculates its active time threshold T̃(i). If TC(t) = 1(r), proceed to Step 3; else proceed to Step 4.
- **Step 3**: Initializing the network topology (initialization phase).
- **Step 4**: Each node adapts its power level and updates the network topology (adaptation phase, Algorithm 1).
- **Step 5**: If the residual energy of node i falls below Ẽ(i), it triggers other nodes in its correlation area within T̃(i).
- **Step 6**: If time elapses to reach T̃(i), node i triggers other nodes in its correlation area.
- **Step 7**: Repeat Steps 2–6 until TC(t) = TC(t)_max.
*(Section IV.C)*

## Complexity Analysis

- **Theorem 2**: the complexity of TC-GSC is O(m × TC(t)_max), where m is the number of active nodes and TC(t)_max is the maximum control time. *(Section IV.D)*
- Justification: the dominant computation is Algorithm 1 in Step 4; each node's best response converges to a minimum power level maintaining connectivity, and after m iterations the power vector converges to the NE p* = (p_1*,...,p_m*). Since the topology is reconfigured adaptively up to TC(t)_max times, the overall complexity scales as O(m × TC(t)_max). *(Section IV.D)*

## Experimental Results

- Simulations were run in a 500×500 m² area with 40–200 nodes (MATLAB 2017b), comparing TC-GSC against DIA, TCLE, and VGEB. *(Section V)*
- Grid search over α, β ∈ [1,10] found that average transmitting power is minimized at α = β = 1.5, which was used in subsequent experiments. *(Section V.A)*
- Across node counts from 40 to 200, TC-GSC showed transmitting power slightly higher than DIA but lower than TCLE and VGEB, lower node degree than TCLE and VGEB, and the smallest average shortest-path hop count among all four algorithms. *(Section V.B, Fig. 3a–c)*
- TC-GSC achieved the smallest variance in residual energy across nodes, indicating the most even energy distribution, and the longest network lifetime (defined as the time until the first node exhausts its energy). *(Section V.B, Fig. 3d–e)*
- Over time (200 nodes, TC(t) tracked across rounds), TC-GSC's transmitting power and node degree increased gradually but remained lower than TCLE and VGEB; its average shortest-path hop count remained consistently the smallest. *(Section V.C, Tables II–IV)*
- Residual energy distribution snapshots (at TC(t) = 1×10²r) showed TC-GSC's residual energy values relatively concentrated compared to DIA, TCLE, and VGEB, which the authors attribute to TC-GSC's node-scheduling mechanism absent in TCLE and VGEB. *(Section V.D, Fig. 4)*

## Pipeline Overview

```
Partition correlation regions
            |
            v
  Select active node set A
   (highest residual energy per cell)
            |
            v
   Initialization phase
 (build neighbor lists via NBM/NRM)
            |
            v
    Adaptation phase
 (Algorithm 1: game-based power
   selection until NE is reached)
            |
            v
   Maintenance phase
 (check residual-energy and
    active-time triggers)
            |
      +-----+------+
      |            |
   triggered    not triggered
      |            |
      v            |
 Reselect active   |
 node; TC(t) += 1  |
      |            |
      +-----> back to "Select active node set A"
                   |
                   v
        TC(t) = TC(t)_max ? --> Stop
```

*(Composed from Sections III.B.1–IV.C, reflecting Steps 1–7 of the algorithm summary)*

## Network Model Constraint Check

- **Unidirectional vs. bidirectional links** — *Deviates.* The paper explicitly requires bidirectional links only: "All links in graph H are bi-directional," and the connectivity condition min{p_m,p_n} ≥ p^th/G_mn is symmetric. *(Section III.A.2, Eq. 4–5)* Unidirectional connectivity is not modeled as valid.
- **Bounded transmission power [p_min, p_max]** — *Partially matches.* An explicit upper bound p_i^max is used throughout (initialization at p_i^max, discretized strategy set). *(Sections III.A.2, IV.A–B)* The discretized set terminates at p_i^η = p_i^min, but the paper does not assign p_min a fixed non-zero numerical floor; it is implicitly bounded below by 0.
- **Multi-hop communication** — *Matches.* h-hop neighbor discovery via NBM/NRM messaging, and utility terms depending explicitly on h-hop neighbor sets (NH_k^h), confirm multi-hop support. *(Sections III.B.2, IV.A)*
- **4-component energy model (sensing, processing, transmitting, receiving)** — *Does not match.* The paper's cost function c_k(p_k(t)) depends only on transmitting power and residual energy (Eq. 12). Sensing, processing, and receiving power are not modeled as separate components anywhere in the paper.