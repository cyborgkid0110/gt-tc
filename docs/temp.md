## Clustering Phase: Adaptive Cluster Radius — Equation Walkthrough

*(From Section IV-B)*

The cluster radius $R(i)$ determines how large a cluster node $i$ would form if selected as a CH. It is composed of three multiplicative components, each correcting for a different aspect of network state:

$$R(i) = (1 + \beta_i)\alpha_i R_0$$

The best way to understand this is to build it from the inside out — starting with $R_0$, then $\alpha_i$, then $\beta_i$.

---

### Component 1: Datum Cluster Radius $R_0$

$$R_0 = \sqrt{\frac{R_m^2 E_{th}}{|N_a| P E_0}}$$

This is the **network-wide baseline** radius — a single value shared by all nodes at a given round. It answers the question: *given the current network state, what is the "average" appropriate cluster radius?*

Each term plays a specific role:

- $R_m$ is the radius of the entire monitoring area. $R_m^2$ appears because cluster area scales with the square of radius — the formula is essentially normalising cluster area relative to monitoring area.
- $P$ is the target fraction of nodes that should become CHs (e.g., $P = 0.05$ means 5% of nodes are CHs). The expected number of clusters is $P \cdot |N_a|$, and each cluster should cover $1/(P \cdot |N_a|)$ of the total area.
- $E_{th}$ is the **median** residual energy of all alive nodes $N_a$. As the network ages and nodes lose energy, $E_{th}$ falls, causing $R_0$ to shrink. This is the mechanism by which the algorithm **automatically contracts cluster sizes** as the network degrades — fewer alive nodes and lower median energy both reduce $R_0$.
- $E_0$ is the initial node energy, used to normalise $E_{th}$ into a dimensionless ratio.

Intuitively, $R_0$ shrinks over time as $E_{th} \downarrow$ and $|N_a| \downarrow$, reflecting the fact that a degraded network should form smaller, more manageable clusters.

---

### Component 2: Local Density Correction $\alpha_i$

$$\alpha_i = \sqrt{\frac{1}{P N_0(i)}}$$

where $N_0(i)$ is the number of alive nodes within radius $R_0$ centred at node $i$.

$R_0$ is a network-average quantity and does not account for **local node density**. In a region where nodes are densely packed, using $R_0$ as the cluster radius would create enormous clusters with too many members, placing excessive processing and aggregation load on the CH. In a sparse region, it might not capture enough nodes to justify the overhead of forming a cluster at all.

$\alpha_i$ corrects for this. The expected number of nodes in a circle of radius $R_0$ under uniform distribution is approximately $P N_0(i)$ clusters worth of nodes. If $N_0(i)$ is large (dense region), $\alpha_i < 1$ and $R_0$ is scaled down, shrinking the cluster. If $N_0(i)$ is small (sparse region), $\alpha_i > 1$ and $R_0$ is scaled up, expanding the cluster to capture enough members. The square root keeps the correction proportional to radius rather than area.

The combined base radius $\alpha_i R_0$ therefore represents the **locally-corrected baseline** before any energy or distance consideration.

---

### Component 3: Energy–Distance Modulation $(1 + \beta_i)$

$$\beta_i = D_E \frac{f_E(i)}{f_{E_{max}}} + (1-D_E) f_D(i), \quad \beta_i \in (0,1)$$

$(1 + \beta_i)$ is a multiplicative boost always greater than 1, meaning high-energy, well-positioned nodes form clusters larger than the local baseline $\alpha_i R_0$. $\beta_i$ is a weighted blend of two factors — and critically, the blend weight $D_E$ itself adapts to the network state.

#### Energy Dispersion Coefficient $D_E$

$$D_E = \frac{\sum_{i \in N_a}(E(i) - E_{ave})^2}{(E_{max} - E_{min})^2 |N_a|}$$

$D_E$ measures how **heterogeneous** the residual energies are across the network. The numerator is the total squared deviation from the mean (sum of squared differences), while the denominator normalises by the maximum possible spread squared times the number of nodes, bounding $D_E \in [0, 1]$.

- When nodes have **similar residual energies** (early network life or after successful balancing), $D_E \approx 0$, and $\beta_i \approx f_D(i)$ — the distance factor dominates. This makes sense: when energy is uniform, node location becomes the primary differentiator for CH suitability.
- When energies are **highly heterogeneous** (late network life, some nodes nearly dead), $D_E \approx 1$, and $\beta_i \approx f_E(i)/f_{E_{max}}$ — the energy factor dominates. This ensures that only energy-rich nodes form large clusters when resources are scarce.

#### Energy Factor $f_E(i)$

$$f_E(i) = \begin{cases} \frac{E(i)}{E_{th}}, & E(i) \geq E_{th} \\ 0, & E(i) < E_{th} \end{cases}$$

This factor is **zero for any node below the median energy**, acting as a hard gate that prevents low-energy nodes from becoming CHs at all. For nodes above $E_{th}$, $f_E(i)$ scales linearly with residual energy: a node with twice the median energy gets twice the energy factor. Normalised by $f_{E_{max}}$ in $\beta_i$, this becomes a relative score among eligible CH candidates.

#### Distance Factor $f_D(i)$

$$f_D(i) = \begin{cases} 0, & i \in S_1 \\ \frac{1}{2}\left(\cos\frac{d_{iBS} - d_{max}}{R_m - d_{max}}\pi + 1\right), & i \in S_2 \end{cases}$$

where $S_1$ is the region within $d_{max}$ of the BS, and $S_2$ is everything beyond.

Nodes in $S_1$ (very close to BS) receive $f_D(i) = 0$, giving them no distance-based boost to cluster radius. This is deliberate: near-BS nodes already face heavy intercluster forwarding load, so they should form small clusters to conserve energy for relay duties. Giving them a large cluster radius would worsen the energy hole.

For nodes in $S_2$, $f_D(i)$ follows a cosine curve that rises smoothly from 0 at $d_{iBS} = d_{max}$ to 1 at $d_{iBS} = R_m$ (the network boundary). This means far-BS nodes get the largest distance-based boost, encouraging them to form larger clusters. The cosine shape ensures a smooth, gradual transition rather than an abrupt step, avoiding instability near the boundary between $S_1$ and $S_2$.

---

### Maximum Single-Hop Distance $d_{max}$

$$d_{max} = 1.2 d_0, \qquad d_0 = \sqrt{\frac{\varepsilon_{fs}}{\varepsilon_{mp}}}$$

$d_{max}$ appears in both $f_D(i)$ and the ICCNS construction. It is derived by comparing single-hop vs. two-hop communication energy for a linear three-node arrangement (Fig. 4 in the paper). The difference $\Delta E = E_{sh} - E_{mh}$ is a monotonically increasing function of spacing $d$, with $\Delta E = 0$ at $d = 0.6 d_0$. Beyond this point, two-hop is always cheaper than single-hop, so the maximum distance at which single-hop is still sensible is $2 \times 0.6 d_0 = 1.2 d_0$.

---

### How All Components Interact

The figure below summarises how $R(i)$ is assembled and how each component responds to network conditions:---

### Putting It All Together: A Worked Example

Consider two candidate CH nodes at a mid-life network stage:

- **Node A**: high residual energy, located far from the BS, in a sparse region.
- **Node B**: moderate residual energy, located near the BS, in a dense region.

For Node A: $E(i) \geq E_{th}$ so $f_E > 0$; $d_{iBS}$ is large so $f_D \approx 1$; $N_0(i)$ is small so $\alpha_i > 1$; and $R_0$ is the shared baseline. All three components push $R(i)$ upward — Node A forms a large cluster to cover the sparse far-BS area.

For Node B: $f_D(i) = 0$ since it is within $d_{max}$, constraining $\beta_i$; $N_0(i)$ is large so $\alpha_i < 1$; these two factors together push $R(i)$ downward — Node B forms a small cluster, preserving its energy for the heavy intercluster forwarding load it is expected to carry.

This interaction is precisely what the paper means by "dynamically adapting cluster size to node energy and distribution."