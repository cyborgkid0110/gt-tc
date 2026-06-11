"""Projected-PSO node placement maximising coverage under a BS-connectivity
constraint, with a hard connectivity-repair post-process.

Decision variables are the N node positions (2*N continuous dims). Particles are
warm-started inside the deployable area and projected back into it after each
update, so they never leave the feasible region. Fitness rewards coverage and
penalises disconnection; a final repair guarantees a BS-connected layout.
"""
import numpy as np
import yaml


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _bs_component_seen(positions, bs_pos, r_conn):
    """Boolean mask (len N+1, index 0 = BS) of the BS-rooted component."""
    n = len(positions)
    pts = np.vstack([np.asarray(bs_pos, dtype=float), positions])  # index 0 = BS
    seen = np.zeros(n + 1, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        nbrs = np.nonzero((d <= r_conn) & ~seen)[0]
        seen[nbrs] = True
        stack.extend(nbrs.tolist())
    return seen


def is_connected_to_bs(positions, bs_pos, r_conn):
    """True iff every node is in the BS-rooted component of the r_conn graph."""
    return bool(_bs_component_seen(positions, bs_pos, r_conn)[1:].all())


def _coverage_fraction(positions, targets, coverage_radius):
    if len(targets) == 0:
        return 0.0
    covered = np.zeros(len(targets), dtype=bool)
    for x, y in positions:
        d = np.hypot(targets[:, 0] - x, targets[:, 1] - y)
        covered |= d <= coverage_radius
    return float(covered.mean())


def _disconnected_fraction(positions, bs_pos, r_conn):
    seen = _bs_component_seen(positions, bs_pos, r_conn)
    return float((~seen[1:]).mean())


def _fitness(positions, targets, coverage_radius, bs_pos, r_conn, lam):
    return (_coverage_fraction(positions, targets, coverage_radius)
            - lam * _disconnected_fraction(positions, bs_pos, r_conn))


def _project_all(region, flat):
    """Project a flat (2N,) vector of positions back into the deployable area."""
    pos = flat.reshape(-1, 2)
    return np.array([region.project(p) for p in pos], dtype=float).ravel()


def _repair_connectivity(positions, region, bs_pos, r_conn, max_iter=1000):
    """Snap disconnected nodes toward the BS component until all are connected."""
    pos = positions.copy()
    for _ in range(max_iter):
        seen = _bs_component_seen(pos, bs_pos, r_conn)
        if seen[1:].all():
            return pos
        n = len(pos)
        anchors = np.vstack([np.asarray(bs_pos, dtype=float), pos])
        comp_pts = anchors[seen]              # connected component (incl. BS)
        # pick the disconnected node nearest the component, pull it toward its
        # nearest component point to just within r_conn, then project to region.
        disc_idx = [j for j in range(n) if not seen[j + 1]]
        best = None
        for j in disc_idx:
            d = np.hypot(comp_pts[:, 0] - pos[j, 0], comp_pts[:, 1] - pos[j, 1])
            k = int(np.argmin(d))
            if best is None or d[k] < best[2]:
                best = (j, comp_pts[k], d[k])
        j, anchor, _dist = best
        direction = (pos[j] - anchor)
        norm = np.hypot(*direction) or 1.0
        target = anchor + direction / norm * (0.9 * r_conn)
        pos[j] = np.array(region.project(target))
    raise RuntimeError(
        "Could not repair connectivity — N too small or region too sparse for "
        f"r_conn={r_conn}.")


def place_nodes(region, num_nodes, radius, seed, config_path):
    """Place ``num_nodes`` to maximise coverage while staying connected.

    A single ``radius`` is the deployment's communication range *and* coverage
    radius: a target is covered within ``radius`` of a node, and two nodes (or a
    node and the BS) are connected when within ``radius`` of each other. This is
    a pure deployment concern — it is independent of the benchmark's energy model
    and physical comm range.
    """
    cfg = _load_cfg(config_path)
    rng = np.random.default_rng(seed)
    targets = region.coverage_targets(cfg['target_spacing'])
    lam = cfg['penalty_lambda']

    dim = 2 * num_nodes
    P = cfg['pso_particles']

    # warm-start particles inside the deployable area
    X = np.array([region.sample_inside(num_nodes, rng).ravel() for _ in range(P)])
    V = np.zeros((P, dim))

    pbest = X.copy()
    pbest_fit = np.array([
        _fitness(x.reshape(-1, 2), targets, radius, region.bs_pos, radius, lam)
        for x in X])
    g = int(np.argmax(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = pbest_fit[g]

    w, c1, c2 = cfg['pso_w'], cfg['pso_c1'], cfg['pso_c2']
    for _ in range(cfg['pso_iters']):
        r1 = rng.random((P, dim))
        r2 = rng.random((P, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        for i in range(P):
            X[i] = _project_all(region, X[i])
            fit = _fitness(X[i].reshape(-1, 2), targets, radius,
                           region.bs_pos, radius, lam)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest = X[i].copy()

    positions = gbest.reshape(-1, 2)
    positions = _repair_connectivity(positions, region, region.bs_pos, radius)
    return positions
