"""Re-simulate clustering runs with an instrumented collector to measure
hop count over *delivering CMs only* (CHs excluded). Writes JSON to scratchpad.
Nothing in results/ is touched.
"""
import json, os, sys, time
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np

import benchmark as B
from metrics import MetricsCollector

OUT = sys.argv[1]
ALGO, TAG, SEED = sys.argv[2], sys.argv[3], int(sys.argv[4])

_orig = MetricsCollector.record_round

def record_round(self, net, t, family_extras=None, routing_tree=None):
    _orig(self, net, t, family_extras, routing_tree)
    if not hasattr(self, 'cm_hop'):
        self.cm_hop, self.cm_hop_strict, self.ch_hop = [], [], []
    tree = routing_tree if routing_tree is not None else net.build_routing_tree()
    by_id = {s.id: s for s in net.sensors}
    cm, cms, ch = [], [], []
    for nid, info in tree.items():
        if not info.get('delivers', True):
            continue
        s = by_id.get(nid)
        if s is None:
            continue
        h = info['depth'] + 1.0
        if s.is_ch:
            ch.append(h)
        else:
            cm.append(h)                          # all non-CH deliverers
            if getattr(s, 'ch_belong', None) is not None:
                cms.append(h)                     # true cluster members only
    self.cm_hop.append(float(np.mean(cm)) if cm else None)
    self.cm_hop_strict.append(float(np.mean(cms)) if cms else None)
    self.ch_hop.append(float(np.mean(ch)) if ch else None)

MetricsCollector.record_round = record_round

from main import build_network, make_algo
B._disable_plotting()

t0 = time.time()
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    net = build_network(scenario=B.scenario_csv(TAG, SEED))
    algo = make_algo(ALGO, net, dict(max_rounds=B.MAX_ROUNDS, plot_period=10**9))
    if ALGO == 'GT2' and TAG in B.GT2_OVERRIDES:
        B._apply_gt2_override(algo, net, B.GT2_OVERRIDES[TAG])
    algo.run()
m = algo.metrics
json.dump({'algo': ALGO, 'deployment': TAG, 'seed': SEED,
           'secs': time.time() - t0,
           'fnd': m.summary().get('fnd'),
           'avg_hop': m.avg_hop,
           'cm_hop': getattr(m, 'cm_hop', []),
           'cm_hop_strict': getattr(m, 'cm_hop_strict', []),
           'ch_hop': getattr(m, 'ch_hop', []),
           'delivered': m.delivered, 'generated': m.generated}, open(OUT, 'w'))
print(f'{ALGO} {TAG} s{SEED}: {time.time()-t0:.1f}s rounds={len(m.avg_hop)}')
