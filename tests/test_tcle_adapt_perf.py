"""Regression guard for TCLE._adapt() performance.

Root cause (2026-05-31): _adapt() called the 145 ms Fiedler-value eigensolve
(_compute_algebraic_connectivity) ~2200 times per pass — once for the current
state and once for each of the kappa trial powers of all 200 sensors — even when
a trial power loses no link and therefore cannot change the graph (and hence
cannot change phi = 1[lambda2 > epsilon]). At ~320 s/pass over hundreds of
passes, the first round never finished.

Fix: when a trial power loses no link, the topology is identical, so phi is
reused (closed-form cost-only delta) with zero graph work; lambda2 is recomputed
only for the rare link-losing trials, and the current-state phi is cached and
invalidated only when an accepted move actually prunes a link.

This test asserts (a) _adapt() finishes quickly on the real 200-node network,
(b) connectivity is preserved (lambda2 > epsilon), and (c) the number of
eigensolves is far below the naive per-pass count.

Run:  conda run -n base python tests/test_tcle_adapt_perf.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_network
from algos.tcle import TCLE


def main() -> int:
    net = build_network()
    algo = TCLE(net, config_path='config/tcle.yaml',
                max_rounds=3, plot_period=999999)

    # Count eigensolves without altering behaviour.
    calls = {'n': 0}
    orig_lambda2 = algo._compute_algebraic_connectivity

    def counted_lambda2():
        calls['n'] += 1
        return orig_lambda2()

    algo._compute_algebraic_connectivity = counted_lambda2

    algo._initialize_topology()

    t0 = time.perf_counter()
    algo._adapt()
    dt = time.perf_counter() - t0

    lam = orig_lambda2()
    n_calls = calls['n']

    print(f'_adapt finished in {dt:.2f}s, '
          f'eigensolves={n_calls}, final lambda2={lam:.4f}')

    # Naive implementation did ~2200 eigensolves *per pass*; the fixed version
    # should do dramatically fewer in total across all passes.
    NAIVE_PER_PASS = 2200
    TIME_BUDGET_S = 60.0

    ok = True
    if dt >= TIME_BUDGET_S:
        print(f'FAIL: _adapt took {dt:.1f}s (budget {TIME_BUDGET_S}s)')
        ok = False
    if lam <= algo.epsilon:
        print(f'FAIL: connectivity lost (lambda2={lam:.4f} <= eps={algo.epsilon})')
        ok = False
    if n_calls >= NAIVE_PER_PASS:
        print(f'FAIL: {n_calls} eigensolves >= naive single-pass cost '
              f'{NAIVE_PER_PASS}')
        ok = False

    print('PASS' if ok else 'TEST FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
