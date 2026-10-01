"""Benchmark metrics collection (shared, algorithm-agnostic).

A MetricsCollector records per-round universal-core metrics by reading
NetworkModel state, and derives scalar summary metrics at the end of a run.
Family-specific extras are computed by the two helper functions and merged in
by the caller (BaseAlgorithm._collect_family_metrics).
"""
import numpy as np


def clustering_family_metrics(net):
    """Per-round extras for clustering algorithms: CH count + cluster sizes."""
    alive = [s for s in net.sensors if s.is_alive]
    chs = [s for s in alive if s.is_ch]
    sizes = {ch.id: 0 for ch in chs}
    for s in alive:
        ch = s.ch_belong
        if ch is not None and ch.id in sizes:
            sizes[ch.id] += 1
    return {'ch_count': len(chs), 'cluster_sizes': list(sizes.values())}


def topology_family_metrics(net):
    """Per-round extras for topology-control algorithms.

    avg_degree: mean out-degree (to alive nodes) over alive nodes.
    lambda2: left None here (optional; not all algorithms expose it).

    Note: avg_tx_power is a universal-core metric (see
    MetricsCollector.record_round) and is therefore no longer duplicated here.
    """
    alive_mask = np.array([s.is_alive for s in net.sensors])
    alive_ids = [s.id for s in net.sensors if s.is_alive]
    if alive_ids:
        degrees = [int(net.edges[i][alive_mask].sum()) for i in alive_ids]
        avg_degree = float(np.mean(degrees))
    else:
        avg_degree = 0.0
    return {'avg_degree': avg_degree, 'lambda2': None}


class MetricsCollector:
    """Collects per-round metrics for a single simulation run."""

    def __init__(self, net):
        self.num_nodes = net.num_nodes
        self.initial_total_energy = float(sum(s.e0 for s in net.sensors))

        # per-round universal-core series
        self.rounds = []
        self.alive = []
        self.total_energy = []
        self.energy_std = []
        self.delivered = []
        self.generated = []
        self.avg_hop = []
        self.avg_tx_power = []
        # family-extra series: metric_name -> list (one entry per recorded round)
        self.family = {}

        self._summary = None

    def record_round(self, net, t, family_extras=None, routing_tree=None):
        """Append one round's metrics. Call once per completed round.

        Delivery is connectivity-based: each alive node generates one packet
        per round (generated = alive count); a packet is delivered iff that
        node has a path to the base station (delivered =
        len(net.build_routing_tree())).  PDR is therefore the fraction of
        alive nodes still connected to the sink — it stays ~1.0 until the
        topology partitions.
        """
        alive_sensors = [s for s in net.sensors if s.is_alive]
        alive = len(alive_sensors)
        energies = [s.e_res for s in alive_sensors]
        total_e = float(sum(energies))
        e_std = float(np.std(energies)) if alive >= 2 else 0.0

        tree = routing_tree if routing_tree is not None \
            else net.build_routing_tree()
        delivering = [info for info in tree.values()
                      if info.get('delivers', True)]
        delivered = len(delivering)
        # avg_hop: mean depth over *delivering* nodes, +1 for the final hop to
        # the BS. None when nothing reaches the BS (undefined, not 0 hops).
        avg_hop = (float(np.mean([info['depth'] for info in delivering])) + 1.0
                   if delivering else None)
        # avg_tx_power: universal-core mean transmit power over alive nodes.
        avg_tx_power = (float(np.mean([s.power for s in alive_sensors]))
                        if alive_sensors else 0.0)

        self.rounds.append(t)
        self.alive.append(alive)
        self.total_energy.append(total_e)
        self.energy_std.append(e_std)
        self.delivered.append(delivered)
        self.generated.append(alive)
        self.avg_hop.append(avg_hop)
        self.avg_tx_power.append(avg_tx_power)

        if family_extras:
            for k, v in family_extras.items():
                self.family.setdefault(k, []).append(v)

    def finalize(self):
        """Derive scalar summary metrics. Returns the summary dict."""
        n = self.num_nodes
        fnd = next((r for r, a in zip(self.rounds, self.alive) if a < n), None)
        hnd = next((r for r, a in zip(self.rounds, self.alive)
                    if a <= n / 2), None)
        lnd = self.rounds[-1] if self.rounds else None

        total_delivered = int(sum(self.delivered))
        total_generated = int(sum(self.generated))
        per_round_pdr = [d / g for d, g in zip(self.delivered, self.generated)
                         if g > 0]
        mean_pdr = float(np.mean(per_round_pdr)) if per_round_pdr else 0.0
        cumulative_pdr = (total_delivered / total_generated
                          if total_generated > 0 else 0.0)

        final_energy = (self.total_energy[-1] if self.total_energy
                        else self.initial_total_energy)
        energy_drained = self.initial_total_energy - final_energy
        energy_per_packet = (energy_drained / total_delivered
                             if total_delivered > 0 else None)
        mean_energy_std = (float(np.mean(self.energy_std))
                           if self.energy_std else 0.0)
        _hops = [h for h in self.avg_hop if h is not None]
        mean_avg_hop = float(np.mean(_hops)) if _hops else 0.0
        mean_avg_tx_power = (float(np.mean(self.avg_tx_power))
                             if self.avg_tx_power else 0.0)

        self._summary = {
            'fnd': fnd,
            'hnd': hnd,
            'lnd': lnd,
            'total_delivered': total_delivered,
            'total_generated': total_generated,
            'mean_pdr': mean_pdr,
            'cumulative_pdr': cumulative_pdr,
            'energy_drained': energy_drained,
            'energy_per_packet': energy_per_packet,
            'mean_energy_std': mean_energy_std,
            'mean_avg_hop': mean_avg_hop,
            'mean_avg_tx_power': mean_avg_tx_power,
        }
        return self._summary

    def summary(self):
        """Scalar summary dict (finalizes lazily if needed)."""
        if self._summary is None:
            self.finalize()
        return self._summary

    def time_series(self):
        """Per-round series as a JSON-serialisable dict."""
        return {
            'rounds': self.rounds,
            'alive': self.alive,
            'total_energy': self.total_energy,
            'energy_std': self.energy_std,
            'delivered': self.delivered,
            'generated': self.generated,
            'avg_hop': self.avg_hop,
            'avg_tx_power': self.avg_tx_power,
            'family': self.family,
        }
