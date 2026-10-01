from abc import ABC, abstractmethod

from model import NetworkModel
from metrics import (
    MetricsCollector,
    clustering_family_metrics,
    topology_family_metrics,
)


class BaseAlgorithm(ABC):
    """Abstract base for all WSN topology-control algorithms."""

    # Subclasses set this to 'clustering' or 'topology' to opt into
    # family-specific per-round metrics. None => universal core only.
    family = None

    def __init__(self, net: NetworkModel, config_path: str,
                 max_rounds: int = 50000, plot_period: int = 1000):
        self.net = net
        self.max_rounds = max_rounds
        self.plot_period = plot_period

        # simulation bookkeeping
        self.t = 0
        self.t_no_dead = None
        self.dead_nodes = 0

        # metrics collection (shared, algorithm-agnostic)
        self.metrics = MetricsCollector(net)
        self._routing_tree = None   # set by _run_round; passed to record_round

    @abstractmethod
    def _run_round(self) -> bool:
        """Execute one simulation round. Return False when network is fully dead."""
        ...

    def run(self):
        """Main simulation loop."""
        while self.t < self.max_rounds:
            self._routing_tree = None
            ok = self._run_round()
            if not ok:
                break
            # record before incrementing t so fnd matches t_no_dead's convention
            self.metrics.record_round(self.net, self.t,
                                      self._collect_family_metrics(),
                                      routing_tree=self._routing_tree)
            self.t += 1

        self.metrics.finalize()

        print(f'Iterations without dead node: {self.t_no_dead}')
        print(f'Iterations without alive node: {self.t}')

    def _collect_family_metrics(self) -> dict:
        """Family-specific per-round extras, dispatched on `self.family`."""
        if self.family == 'clustering':
            return clustering_family_metrics(self.net)
        if self.family == 'topology':
            return topology_family_metrics(self.net)
        return {}

    def _charge_cluster_maintenance(self):
        """Build the cluster routing tree, stash it, and charge energy to every
        alive node along the CM->relay->CH->backbone->BS path (forward-only
        members, CH aggregation), tracking deaths. Returns the tree.
        """
        net = self.net
        tree = net.build_cluster_routing_tree()
        self._routing_tree = tree
        costs = net.compute_cluster_maintenance_costs(tree)
        for s in net.sensors:
            if not s.is_alive or s.id not in costs:
                continue
            cost = costs[s.id]
            if s.is_ch:
                s.c_ch = cost
            else:
                s.c_cm = cost
            s.e_res -= cost
            if s.e_res <= 0:
                self._track_death(s)
        return tree

    def _track_death(self, sensor):
        """Increment dead count and record first-death round."""
        self.dead_nodes += 1
        if self.t_no_dead is None:
            self.t_no_dead = self.t
        # print('Dead nodes:', self.dead_nodes)
