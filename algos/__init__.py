from abc import ABC, abstractmethod
import yaml

from model import NetworkModel


class BaseAlgorithm(ABC):
    """Abstract base for all WSN topology-control algorithms."""

    def __init__(self, net: NetworkModel, config_path: str):
        self.net = net

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.max_rounds = cfg['max_rounds']
        self.plot_period = cfg['plot_period']

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
        print('Dead nodes:', self.dead_nodes)
