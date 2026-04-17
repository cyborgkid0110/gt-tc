# Backgrounds

## Game Theory

Game theory uses mathematical tools to solve decision-making problems. The non-cooperative model deals with the interactions among individual decision-makers [38]. A model is called a game, and the decision-maker is called a player.

A strategic non-cooperative game L = ⟨B, S, u⟩ consists of three elements:

- **Player set B**: B = {b_1, b_2, ..., b_n}, where n is the number of players in the game.
- **Strategy set S**: s ∈ S = (S_1, S_2, ..., S_n), where S denotes the Cartesian product of i-th player's action and s_i ∈ S_i is the i-th player's strategy over the set of its possible strategies S_i. Usually, we denote a strategy profile s = (s_i, s_{-i}), where s_i is the i-th player's strategy, and s_{-i} denotes the strategy of the other n-1 players.
- **Utility function**: For each player b_i ∈ B, utility function u_i : S → ℝ models the player's preferences over strategy profiles. U = {u_1, u_2, ..., u_n} : S → ℝ denotes the vector of such utility functions.

Nash equilibrium (NE) is an important concept in game theory. A strategy profile s* = (s_1*, s_2*, ..., s_n*) is a NE if no player can increase its utility by unilaterally changing its strategy, i.e., for every player b_i ∈ B and every strategy s_i ∈ S_i, we have u_i(s_i*, s_{-i}*) ≥ u_i(s_i, s_{-i}*).

This state is called pure strategy Nash equilibrium. In this state, players are in a best response correspondence with one another. However, a fundamental limitation of pure strategies is that a stable equilibrium does not necessarily exist in all games. Moreover, in some games such as Zero-sum games, pure strategies can be exploited easily by opponents. Therefore, mixed strategy games are introduced to ensure the uncertainty of players' behaviour.

- **Strategy distribution**: P_i(S_i) = {p_i | p_i ≥ 0, Σ_{s_i ∈ S_i} p_i(s_i) = 1}, where p_i(s_i) is the probability that player b_i chooses strategy s_i.
- **Mixed strategy profile**: P = (P_1, P_2, ..., P_n).
- **Expected utility**: For each player b_i ∈ B, the expected utility function U_i : P → ℝ is defined as U_i(P) = Σ_{s ∈ S} u_i(s) Π_{j=1}^{n} p_j(s_j), where s = (s_1, s_2, ..., s_n) is a strategy profile.

A mixed strategy profile P* = (P_1*, P_2*, ..., P_n*) is a mixed strategy NE if no player can increase its expected utility by unilaterally changing its strategy distribution, i.e., for every player b_i ∈ B and every strategy distribution P_i ∈ P_i(S_i), we have U_i(P_i*, P_{-i}*) ≥ U_i(P_i, P_{-i}*).

---

## WSN Modeling

### System Model

#### Network Topology

We model the Wireless Sensor Network (WSN) as a directed graph G = (V, E). Here, V = {v_1, v_2, ..., v_N} represents the set of N sensor nodes distributed in a two-dimensional Euclidean plane, and E represents a set of edges denoting the communication links between these nodes.

An **adjacency matrix** L represents the connectivity of G network. Let l_{ij} be denoted as the connection from node v_i to node v_j. The values of the adjacency matrix are defined as follows:

```
l_{ij} = 1,  if (v_i, v_j) ∈ E
         0,  otherwise
```

As the network is directional (due to potential heterogeneity in transmission power), the adjacency matrix E is not necessarily symmetric, meaning l_{ij} = 1 does not imply l_{ji} = 1.

The connectivity l_{ij} is determined by the transmitting power of node v_i, the physical distance between them, and the signal propagation characteristics. The Friis free-space propagation model is adapted to determine the received signal strength. The received power P_r at a receiver node v_j from a transmitter node v_i is given by:

```
P_r(d_{ij}) = P_t * G_t * G_r * (λ / (4π * d_{ij}))²
```

where:
- P_t ∈ [0, p_max] is the transmission power of node v_i.
- G_t and G_r are the antenna gains of the transmitter and receiver, respectively.
- λ is the wavelength of the carrier frequency.
- d_{ij} is the Euclidean distance between node v_i and node v_j.

A directed edge (v_i, v_j) ∈ E if and only if the received power at v_j is larger than a common signal capture threshold P_th. Solving the Friis equation for distance, we define the maximum transmission range R_tx as:

```
R_tx = (λ / 4π) * sqrt(P_t * G_t * G_r / P_th)
```

That is, the set of edges E is defined formally as:

```
E = { (v_i, v_j) | v_i, v_j ∈ V, i ≠ j, and d_{ij} ≤ R_tx }
```

### Local Network Definition (k-hop Neighborhood)

We define the local network of a node based on multi-hop connectivity. Let the connection from node v_i with transmitting power p_i to node v_j within k hops be denoted l_{ij}^k(p_i). For directed connection (1-hop), l_{ij}^1(p_i) = l_{ij}(p_i). If node v_j is also connected to node v_u, then l_{iu}^2(p_i) = l_{ij}(p_i) · l_{ju}^1(p_j). Generalized to k-hop connection:

```
l_{ij}^k(p_i) = 1,  if l_{ij}^k = l_{iu}(p_i) · l_{uj}^{k-1}(p_u)
                0,  otherwise
```

The **local network** of node v_i, denoted as G_local(v_i), is defined as the induced subgraph of G formed by node v_i and its k-hop neighbors:

```
G_local(v_i) = (V'_i, E'_i)
```

where:
- V'_i = { v_i; v_j ∈ V | l_{ij}^{k'}(p_i) = 1, ∀k' ∈ [1, k] }
- E'_i = { (v_u, v_j) ∈ E | u, j ∈ V'_i }

---

## Energy Model

The energy consumption of a sensor node is related to many factors. The main factors that directly influence the energy consumption of sensor nodes are discussed below. The energy consumption of a sensor node can be divided into three parts: sensing energy consumption, data processing energy consumption, and communication energy consumption.

### Sensing Energy Consumption

When a sensor node works, it senses data from the environment. The energy consumption of sensing is related to the type of sensor and the frequency of sensing. In general, the energy consumption of sensing can be defined as:

```
E_sense = L(S_i) · I(S_i) · V_dd · t_sense
```

where I(S_i) is the current of sensor node S_i, V_dd is the supply voltage, and t_sense is the time duration of sensing and collecting L(S_i) bits of data.

### Processing Energy Consumption

When the sensor node reads and stores data, it also consumes energy. The energy consumption can be expressed as:

```
E_process = (1/8) × L(S_i) × V_dd × (I_write × t_write + I_read × t_read)
```

where I_write and I_read are the current of writing and reading data from memory, and t_write and t_read are the time duration of writing and reading per bit of data, respectively.

### Communication Energy Consumption

The energy consumption of sensors mainly comes from communication. The communication energy consumption includes the energy consumption of transmitting and receiving data.

The energy consumption to transmit a package of n bits over a distance d is defined by:

```
E_transmit(n, d) = E_tc(n) + E_amp(n, d)
                 = n·E_trans + n·ε_amp·d^α
```

where E_tc(n) is the energy that the radio circuit needs to consume in order to process n bits, E_amp(n, d) is the energy needed by the radio amplifier circuit to send n bits d meters, E_trans is the energy needed to process a single bit by the radio transmission circuits, and α is the path-loss exponent. ε_amp is the transceiver's energy dissipation:

```
ε_amp = [ (S/N_r) · NF_RX · N_0 · BW · (4π/λ)^γ ] / [ G_ant · η · R_bit ]
```

where S/N_r is the signal to noise ratio at the receiver, NF_RX is the receiver noise figure, N_0 is the noise power spectral density, BW is the channel noise bandwidth, λ is the wavelength in meters, G_ant is the antenna gain, η is the transmitter efficiency, and R_bit is the channel data rate in bits per second.

The energy consumption to receive n bits is:

```
E_receive(n) = n × E_elec
```

CH performs aggregation and compression before sending data to the sink. Energy expenditure E_agg in aggregating k packets of m-bit length is:

```
E_agg = k × m × ε_agg
```

### Total Energy Consumption for Sensor Node

The total energy consumption of a sensor node S_i is:

```
C_i = E_sense + E_process + E_transmit + E_receive + E_agg
```

---

# Proposed Game-Theoretic Framework

In this section, two game-theoretic frameworks are proposed. The first framework is a non-cooperative mixed-strategy game designed to select Cluster Heads (CHs), thereby forming sensor clusters. The second framework is a non-cooperative pure-strategy game utilized to adjust the transmission power of sensor nodes within each cluster, aiming to optimize energy consumption and prolong network lifetime.

## Clustering Game

In this game, Cluster Heads (CHs) are self-elected from the set of sensor nodes, subject to the constraint that there is at least one CH in the network. Neighboring sensor nodes select the nearest CH to join its cluster. The CHs are responsible for collecting data from cluster members, aggregating it, and forwarding it to the sink node, either directly or via other CHs.

### Game Formulation

The clustering game can be formally defined as a non-cooperative mixed Nash game G_c = ⟨N, S_c, u_c⟩, where:

- **Player set**: N = {1, 2, ..., n} represents the set of all sensor nodes in the network.

- **Strategy set**: Each sensor node i ∈ N has two pure strategies:
  ```
  S_i^c = {CH, CM}
  ```
  where CH (cluster head) and CM (cluster member) represent the two possible roles a node can assume. The overall strategy profile is s^c = (s_1^c, s_2^c, ..., s_n^c), where s_i^c ∈ S_i^c.

- **Mixed strategy**: Let p_i ∈ [0, 1] denote the probability that node i chooses to be a CH, and (1 - p_i) denote the probability of being a CM. The mixed strategy profile is P = (P_1, P_2, ..., P_n).

- **Utility function**: The utility function for node i is defined as:
  ```
  u_i(s^c) = u_CH - C_CH,        if s_i^c = CH
             u_CM - C_CM,        if s_i^c = CM and ∃ s_{-i} = CH
             0,                  if ∀ s = CM
  ```
  where:
  - u_CH is the benefit of being a cluster head (e.g., reduced communication distance to other nodes);
  - u_CM is the benefit of being a cluster member;
  - C_CH is the cost of being a cluster head, including data aggregation and routing overhead;
  - C_CM is the cost of being a cluster member, including transmission cost to the nearest cluster head.

### Node Selection and Cluster Formation

The clustering process follows these rules:

1. Each node independently decides its role (CH or CM) based on its mixed strategy probability p_i.
2. At least one cluster head must be selected in the network. If no node chooses to be a CH, a penalty is applied to all nodes to encourage CH formation.
3. Once cluster heads are selected, each cluster member node j joins the nearest cluster head i that minimizes the distance d(i, j).
4. Cluster heads are responsible for:
   - Collecting data from cluster members;
   - Aggregating the collected data;
   - Forwarding data to the sink node either directly or through other cluster heads.

### Nash Equilibrium in Clustering Game

In a symmetrical clustering game, where NE exists with symmetrical mixed strategies, the probability P_0 that a node declares itself as CH is:

```
P_0 = 1 - ( (c_CH - c_CM) / (ρ - c_CM) )^(1/(N-1))
```

---

## Transmission Power Control Game

After the clustering game, a set of cluster members in each cluster is obtained. Each cluster member must then optimize its transmission power to minimize energy consumption while maintaining reliable communication within its cluster. This is formulated as a second non-cooperative game among the cluster heads.

### Game Formulation

The transmission power control game can be formally defined as a non-cooperative game G_p = ⟨M, S_p, u_p⟩, where:

- **Player set**: M = {i_1, i_2, ..., i_m} represents the set of all cluster members in the cluster, where m is the number of CMs.

- **Strategy set**: Each cluster member i ∈ M can adjust its transmission power within a continuous range:
  ```
  S_i^p = {P_i | P_min ≤ P_i ≤ P_max}
  ```
  where P_min and P_max are the minimum and maximum transmission power levels respectively. The strategy profile is s^p = (P_1, P_2, ..., P_m), where P_i ∈ S_i^p.

- **Utility function**: The utility function for cluster member i is defined as:
  ```
  u_i(s^p) = f_i(p_i, p_{-i}) · [ f_e(i) + f_pr(i) ] - c_i(p_i)
              \_____neighbor______/  \______benefit_____/   \_cost_/
  ```
  where:
  - f_i(p_i, p_{-i}) ∈ {0,1}: equals 1 if node i can establish connections with its neighbor nodes (bi-directional or directional), 0 otherwise.
  - f_e(i) is the energy-balance benefit from the topology:
    ```
    f_e(i) = -α · Σ(E_r(i) - E_r_avg^k)² / |V'_i|
    ```
    where E_r(i) is the residual energy of node v_i, E_r_avg^k is the average residual energy of all neighbor nodes of v_i, |V'_i| is the number of neighbor nodes of v_i, and α > 0 is a constant weight factor.
  - f_pr(i) represents the contribution to form the connected topology:
    ```
    f_pr(i) = β · |V'_i|
    ```
  - c_i(p_i) is the cost of node i when choosing power p_i. The lower the residual energy E_r(i), the less likely node v_i is to deplete its valuable energy resources:
    ```
    c_i(p_i) = u · ∫[E_0(i)-E_r(i) to E_0(i)-E_r(i)+p_i(t)·T] g(x) dx
    ```
    where E_0(i) is the initial energy of node v_i, T denotes the unit transmission time satisfying p_i(t)·T ≤ E_r(i), u is sufficiently small so that c_i(p_i(t)) ∈ [0,1], and g(x) is the energy-cost function defined as g(x) = exp(x/10).

### Nash Equilibrium in Power Control Game

The transmission power control algorithm operates in an iterative manner:

1. Each CH i selects a new power by decreasing its power without losing connectivity to its neighbors:
   ```
   P_i = max(P_min, P_i - ΔP_i)
   ```
   where ΔP_i is the power step. The step must be selected small enough such that at most one connection will be deleted.

2. With the new power, each CH i evaluates its utility function u_i(s^p) based on the new power level and the current power levels of other CHs. The power level is accepted if it results in an increased utility; otherwise, the previous power level is retained.

3. The algorithm terminates when no node's power level is selected to reduce its transmitting power further. The final power levels constitute the Nash equilibrium of the transmission power control game.

---

# Methodology

The proposed methodology for the game-theoretic Wireless Sensor Network (WSN) topology control is implemented through a simulation framework that integrates the theoretical models from the Backgrounds and Proposed Game-Theoretic Framework sections. The process operates in iterative rounds, simulating network evolution over time. Each round consists of five phases: initialization, cluster formation, network formation, adaptation, and maintenance.

**Simulation parameters:**
- N = 200 sensor nodes distributed in a 250×250 area
- Initial energy E_0 = 0.5 J per node
- Transmission power range [P_min = 0.01, P_max = 0.08]
- E_elec = 50×10⁻⁹ J/bit, α = 1.5, β = 0.1

## Initialization Phase

Nodes are initially generated with transmitting power p_i ∈ [p_min, p_max]. The node deployment must satisfy connectivity at least when p_i = p_max.

This phase sets the stage for the clustering game by broadcasting advertisement (ADV) messages to identify 1-hop neighbors, populating each node's neighbor list based on distance d_{ij} ≤ R_c.

In the initialization stage, all nodes broadcast an "ADVT" message to collect information about their neighboring nodes. After receiving the "ADVT" message, each node detects the number of its neighboring nodes.

## Cluster Formation Phase

This phase implements the non-cooperative mixed-strategy clustering game G_c = ⟨N, S_c, u_c⟩. Each alive node v_i with neighbors computes its cluster head (CH) probability:

```
P_CH = P_0 = 1 - ( (C_CH - C_CM) / (ρ - C_CM) )^(1/(N-1))
```

adjusted by residual energy. A random draw determines if v_i becomes a CH.

If no CHs are selected, the round restarts to enforce at least one CH. Selected CHs set their power to P_max and broadcast advertising messages. Cluster members (CMs) join the nearest CH based on Euclidean distance d_{ij}, updating their CH affiliation. CHs also identify neighboring CHs for inter-cluster routing. Costs C_CH and C_CM are calculated using the energy model, incorporating sensing, processing, transmission, reception, and aggregation energies.

## Network Formation Phase

Following cluster formation, the network graph is refined to ensure connectivity within and between clusters. Unconnected nodes (those without a CH affiliation) attempt to join clusters by sending join requests to surrounding CMs, incrementally increasing their transmission power by ΔP = 0.0001 until connected or P_max is reached. If connectivity fails, the node remains isolated.

Edges are updated to reflect intra-cluster connections only among nodes in the same cluster, removing extraneous links. The graph is divided into clusters using BFS layering from each CH, forming layered batches per cluster (e.g., layers based on hop distance from CH). This phase ensures the local network G_local(v_i) is defined for each node, with k-hop neighborhoods (up to k=3) for topology control.

## Adaptation Phase

This phase executes the non-cooperative pure-strategy transmission power control game G_p = ⟨M, S_p, u_p⟩ within each cluster, optimizing CM transmission powers iteratively toward Nash equilibrium (NE). For each CM v_i in reversed layer order (starting from outermost layers), a new power P_i' = max(P_min, P_i - ΔP) is tested.

The utility u_i(s^p) = f_i(P_i, P_{-i}) · [f_e(i) + f_pr(i) - c_i(P_i)] is recomputed, where:
- f_i(P_i, P_{-i}) checks bidirectional connectivity in the local k-hop network.
- f_e(i) = -α · Σ(E_r(i) - E_r_avg^k)² / |V'_i| balances energy.
- f_pr(i) = β · |V'_i| rewards connectivity.
- c_i(P_i) = u · ∫ exp(x/10) dx penalizes energy use.

If the new utility exceeds the current one and the local network remains connected (verified via DFS), the power is updated, and the global graph is adjusted. Iteration continues until no further improvements occur, achieving NE.

## Maintenance Phase

In the final phase, data transmission simulates network operation, deducting energy based on roles and depths. For each cluster, the CH consumes C_CH for aggregation and forwarding. CMs in layer l (depth from CH) consume C_CM adjusted by layer depth, including multi-hop forwarding costs.

Residual energies are updated: E_r(i) ← E_r(i) - C_i. Dead nodes are counted if E_r(i) ≤ 0. Plots (e.g., cluster probabilities, transmission powers, directional graphs) are generated every 50 rounds for visualization. The simulation halts after 50,000 rounds or when all nodes are dead, tracking metrics like rounds until first death.

---

# Simulation Results

*(To be added)*
