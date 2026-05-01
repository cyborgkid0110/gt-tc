import random
import numpy as np
from scipy.stats import qmc

from model import Sensor, NetworkModel
from algos.gt2 import GT2
from algos.leach import LEACH
from algos.gtfr import GTFR
from algos.dia_mia import DIAMIA
from algos.tcle import TCLE
from algos.eftcg import EFTCG
from algos.fl_leach_pso import FLLEACHPSO
from algos.sca_levy import SCALEVY
from algos.fc_cra import FCCRA
from algos.ee_tcm import EETCM

########################################################################
# GLOBAL PARAMETERS (shared across all algorithms)
########################################################################

NUM_NODES = 200
AREA = 250
E0 = 0.5                        # initial energy (J)
P_MIN = 0.01
P_MAX = 0.08
P_STEP = 0.0001
WAVE = 0.1224
PTH = 7e-10                     # signal capture threshold
HOP_MAX = 3

# energy model
E_ELEC = 50e-9
E_AGG = 5e-9
M_PKT_S = 20
M_PKT_L = 1000

########################################################################
# NODE GENERATION (Poisson Disk Sampling)
########################################################################

seed = 42
rng = np.random.default_rng(seed)
engine = qmc.PoissonDisk(d=2, radius=30, rng=rng, ncandidates=NUM_NODES,
                         l_bounds=0, u_bounds=AREA * 2)
sample = engine.random(NUM_NODES)

not_generated = NUM_NODES - len(sample)
while not_generated > 0:
    row = np.round(np.random.rand(1, 2) * AREA * 2, 2)
    sample = np.append(sample, row, axis=0)
    not_generated -= 1

sensors = []
for i in range(len(sample)):
    x = float(sample[i, 0] - AREA)
    y = float(sample[i, 1] - AREA)
    sensors.append(Sensor(id=i, x=x, y=y, e0=E0,
                          power=P_MAX / 4,
                          Vpre=random.uniform(2.7, 4.2)))

net = NetworkModel(sensors, AREA,
                   pth=PTH, wave=WAVE, p_min=P_MIN, p_max=P_MAX, p_step=P_STEP,
                   hop_max=HOP_MAX, e_elec=E_ELEC, e_agg=E_AGG,
                   m_pkt_s=M_PKT_S, m_pkt_l=M_PKT_L)

print("Generated done")
print("Possible connectivity:", net.check_potential_connectivity())

########################################################################
# RUN ALGORITHM
########################################################################

ALGORITHM = 'TCLE'  # 'GT2', 'LEACH', 'GTFR', 'DIA', 'MIA', 'TCLE', 'EFTCG-1', 'EFTCG-2', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA', 'EE-TCM'

if ALGORITHM == 'GT2':
    algo = GT2(net, config_path='config/gt2.yaml')
elif ALGORITHM == 'LEACH':
    algo = LEACH(net, config_path='config/leach.yaml')
elif ALGORITHM == 'GTFR':
    algo = GTFR(net, config_path='config/gtfr.yaml')
elif ALGORITHM == 'DIA':
    algo = DIAMIA(net, config_path='config/dia_mia.yaml')
elif ALGORITHM == 'MIA':
    algo = DIAMIA(net, config_path='config/dia_mia.yaml')
    algo.mode = 'MIA'
elif ALGORITHM == 'TCLE':
    algo = TCLE(net, config_path='config/tcle.yaml')
elif ALGORITHM == 'EFTCG-1':
    algo = EFTCG(net, config_path='config/eftcg.yaml')
elif ALGORITHM == 'EFTCG-2':
    algo = EFTCG(net, config_path='config/eftcg.yaml')
    algo.k = 2
elif ALGORITHM == 'FL-LEACH-PSO':
    algo = FLLEACHPSO(net, config_path='config/fl_leach_pso.yaml')
elif ALGORITHM == 'SCA-LEVY':
    algo = SCALEVY(net, config_path='config/sca_levy.yaml')
elif ALGORITHM == 'FC-CRA':
    algo = FCCRA(net, config_path='config/fc_cra.yaml')
elif ALGORITHM == 'EE-TCM':
    algo = EETCM(net, config_path='config/ee_tcm.yaml')

algo.run()
