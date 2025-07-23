import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import qmc
from scipy import integrate
import csv
from plot import directional_wsn_plot
import graph
import cupy as cp

num_nodes = 200
dead_nodes = 0
dead_nodes_t = []
area = 250
xs, ys = (0, 0)

# system parameters
e0 = 1                     # initial energy of the nodes
eth = 20                    # threshold energy
pth = 0.5*pow(10, -9)         # power threshold
hop_max = 2                 # number of neighbor hops
d = 50                      # cell size
Vsta = 3.6                  # standard working voltage
gamma = 5
p_strat = []
p_min = 0.01
p_max = 0.08
p_step = 0.0001
wave = 0.1224

# game 1 parameters
e_mp = 0.0013*pow(10, -12)
e_fs = 10*pow(10, -12)
e_elec = 50*pow(10, -9)
e_agg = 5*pow(10, -9)
d0 = math.sqrt(e_fs/e_mp)
m_pkt_s = 20
m_pkt_l = 500
payoff = 0.02

# game 2 parameters
ALPHA = 1.5     # e_balance_benefit
BETA = 1.5      # ctb_benefit
M = 0.01        # e_cost

def cal_rc(power):
    distance = math.sqrt((power*wave*wave)/(pth*16*math.pi*math.pi))
    # print(distance)
    return distance

t = 0
max_t = 50000
node_dict = {}
network = {
    'edges': None,
    'vertices': [],
}

seed = 42
rng = np.random.default_rng(seed)
radius = 30
engine = qmc.PoissonDisk(d=2, radius=radius, rng=rng, ncandidates=num_nodes, l_bounds=0, u_bounds=area * 2)
sample = engine.random(num_nodes)
not_generated_nodes = num_nodes - len(sample)
sample_gpu = cp.array(sample)

if not_generated_nodes > 0:
    rows = cp.round(cp.random.rand(not_generated_nodes, 2) * area * 2, 2)
    sample_gpu = cp.vstack((sample_gpu, rows))
    not_generated_nodes = 0

for i in range(0, len(sample_gpu)):
    xn, yn = float(sample_gpu[i, 0] - area), float(sample_gpu[i, 1] - area)
    network['vertices'].append((xn, yn))
    
    node_dict[(xn, yn)] = {
        'id': i,
        'neighbors': [],
        'power': p_max / 4,
        'rc': cal_rc(p_max / 4),
        'e_res': e0,
        'Vpre': random.uniform(2.7, 4.2),
        'p0': None,
        'p_ch': None,
        'CH': False,
        'CH_belong': None,
        'CH_neighbors': [],
        'c_ch': 0,
        'c_cm': 0,
        
        'util': None,
        'local_net': None
    }

print("Generated done")

while (t < max_t):
    CH_con = 0
    CH_can = 0
    CH_true = 0
    end_loop = False
    
    # reset network for each round
    network['edges'] = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(0, num_nodes):
        node = network['vertices'][i]
        node_dict[node]['power'] = p_max / 4
        node_dict[node]['rc'] = cal_rc(p_max / 4)
        node_dict[node]['neighbors'] = []
        node_dict[node]['CH_belong'] = None
        node_dict[node]['util'] = None
        node_dict[node]['local_net'] = None
    
    