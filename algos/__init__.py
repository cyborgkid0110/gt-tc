from abc import ABC, abstractmethod

from model import NetworkModel


class BaseAlgorithm(ABC):
    """Abstract base for all WSN topology-control algorithms."""

    def __init__(self, net: NetworkModel, config_path: str,
                 max_rounds: int = 50000, plot_period: int = 1000):
        self.net = net
        self.max_rounds = max_rounds
        self.plot_period = plot_period

        # simulation bookkeeping
        self.t = 0
        self.t_no_dead = None
        self.dead_nodes = 0

    @abstractmethod
    def _run_round(self) -> bool:
        """Execute one simulation round. Return False when network is fully dead."""
        ...

    def run(self):
        """Main simulation loop."""
        while self.t < self.max_rounds:
            ok = self._run_round()
            if not ok:
                break
            self.t += 1

        print(f'Iterations without dead node: {self.t_no_dead}')
        print(f'Iterations without alive node: {self.t}')

    def _track_death(self, sensor):
        """Increment dead count and record first-death round."""
        self.dead_nodes += 1
        if self.t_no_dead is None:
            self.t_no_dead = self.t
        # print('Dead nodes:', self.dead_nodes)
