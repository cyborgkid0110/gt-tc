"""GT2 config loading must coerce numeric params to numbers.

PyYAML's float resolver does NOT recognise scientific notation written
without a decimal point (e.g. ``2e-5``) — it parses it as a *string*. GT2
then crashes on ``self.payoff - c_cm`` (str - float). GT2 must coerce its
numeric config params so any valid numeric spelling works.

Run:  conda run -n base python tests/test_gt2_config.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Sensor, NetworkModel
from algos.gt2 import GT2


def _tiny_net():
    sensors = [Sensor(id=i, x=float(i * 10), y=0.0, e0=0.05,
                      power=2.5e-4 / 4, Vpre=3.0) for i in range(5)]
    return NetworkModel(sensors, 250)


def _gt2_from_config(text):
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return GT2(_tiny_net(), config_path=path,
                   max_rounds=3, plot_period=10 ** 9)
    finally:
        os.unlink(path)


def test_gt2_coerces_dotless_scientific_payoff():
    # 'payoff: 2e-5' is parsed by PyYAML as the STRING '2e-5'.
    algo = _gt2_from_config(
        "payoff: 2e-5\nalpha: 1.5\nbeta: 0.1\nmu: 0.01\nhop_max: 3\n")
    assert isinstance(algo.payoff, float) and math.isclose(algo.payoff, 2e-5)
    assert isinstance(algo.alpha, float) and isinstance(algo.beta, float)
    assert isinstance(algo.mu, float)
    assert isinstance(algo.hop_max, int) and algo.hop_max == 3
    # pushed onto the net for the power-control game
    assert isinstance(algo.net.beta, float) and algo.net.hop_max == 3
    # the exact operation that raised TypeError in the benchmark
    _ = algo.payoff - 1.0


def test_gt2_handles_dotted_and_string_payoff():
    # dotted scientific (already a float) and an explicitly-quoted string both work.
    a1 = _gt2_from_config(
        "payoff: 3.4e-5\nalpha: 2.5\nbeta: 0.3\nmu: 0.05\nhop_max: 3\n")
    assert math.isclose(a1.payoff, 3.4e-5)
    a2 = _gt2_from_config(
        "payoff: '6e-5'\nalpha: 1.5\nbeta: 0.1\nmu: 0.01\nhop_max: 2\n")
    assert isinstance(a2.payoff, float) and math.isclose(a2.payoff, 6e-5)
    assert a2.hop_max == 2


if __name__ == '__main__':
    test_gt2_coerces_dotless_scientific_payoff()
    test_gt2_handles_dotted_and_string_payoff()
    print('GT2 config coercion tests passed')
