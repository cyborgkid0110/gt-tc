"""Smoke test for the parallel sweep runner.

Run:  conda run -n base python tests/test_benchmark_smoke.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark


def test_run_one_writes_json_and_csv():
    d = tempfile.mkdtemp()
    out = benchmark.run_one('LEACH', 'poisson', 1, 40, d, max_rounds=2)
    assert os.path.exists(out), out

    with open(out) as f:
        payload = json.load(f)
    assert payload['algo'] == 'LEACH'
    assert payload['deployment'] == 'poisson'
    assert payload['seed'] == 1
    assert payload['num_nodes'] == 40
    assert 'summary' in payload and 'time_series' in payload
    assert 'fnd' in payload['summary']
    assert 'alive' in payload['time_series']
    print('  run_one JSON OK')

    # second call must skip (idempotent / resumable)
    out2 = benchmark.run_one('LEACH', 'poisson', 1, 40, d, max_rounds=2)
    assert out2 == out
    print('  skip-existing OK')

    benchmark.build_summary_csv(d)
    csv_path = os.path.join(d, 'summary.csv')
    assert os.path.exists(csv_path), csv_path
    with open(csv_path) as f:
        header = f.readline()
    assert 'algo' in header and 'fnd' in header
    print('  summary.csv OK')


if __name__ == '__main__':
    test_run_one_writes_json_and_csv()
    print('\nBenchmark smoke test passed.')
