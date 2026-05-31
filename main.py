import argparse
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
E0 = 0.005                        # initial energy (J)
P_MIN = 3.0e-5
P_MAX = 2.5e-4
P_STEP = 3.0e-6
HOP_MAX = 3

# radio link-budget parameters (Tudose et al. Eq. 7-8)
SNR = 10                          # required SNR (linear, = 10 dB)
NF_RX = 6.31                     # receiver noise figure (linear, = 8 dB)
N0 = 3.98e-21                    # noise PSD (W/Hz), kT at 290 K
BW = 3e6                         # channel bandwidth (Hz)
WAVE = 0.125                     # wavelength (m), c / 2.4 GHz
GAMMA = 2.0                      # path loss exponent
G_ANT = 1.0                      # antenna gain (linear, = 0 dBi)
ETA = 0.30                       # TX PA efficiency
R_BIT = 250e3                    # data rate (bps)

# energy model
E_ELEC = 50e-9
E_AGG = 5e-9
# packet structure (application payload only), all in bits
DATA_PAYLOAD = 32                 # data packet: node id 16 + sensor value 16 (4 B)
AGG_PAYLOAD = 72                  # agg packet: cluster id 16 + count 8 + mean 16 + min 16 + max 16 (9 B)
SENSOR_SAMPLE_BITS = 16           # 16-bit ADC sample (sensing/processing)

# simulation
MAX_ROUNDS = 50000
PLOT_PERIOD = 100

ALGO_CHOICES = [
    'GT2', 'LEACH', 'GTFR', 'DIA', 'MIA', 'TCLE',
    'EFTCG-1', 'EFTCG-2', 'FL-LEACH-PSO', 'SCA-LEVY', 'FC-CRA', 'EE-TCM',
]

########################################################################
# NODE GENERATION
########################################################################

def build_network():
    seed = 42
    rng = np.random.default_rng(seed)
    engine = qmc.PoissonDisk(d=2, radius=30, rng=rng, ncandidates=NUM_NODES,
                             l_bounds=0, u_bounds=AREA * 2)
    sample = engine.random(NUM_NODES)

    not_generated = NUM_NODES - len(sample)
    random.seed(seed)
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
                       snr=SNR, nf_rx=NF_RX, n0=N0, bw=BW,
                       wave=WAVE, gamma=GAMMA, g_ant=G_ANT, eta=ETA, r_bit=R_BIT,
                       p_min=P_MIN, p_max=P_MAX, p_step=P_STEP,
                       hop_max=HOP_MAX, e_elec=E_ELEC, e_agg=E_AGG,
                       data_payload=DATA_PAYLOAD, agg_payload=AGG_PAYLOAD,
                       sensor_sample_bits=SENSOR_SAMPLE_BITS)
    return net

########################################################################
# ALGORITHM INSTANTIATION
########################################################################

def make_algo(name, net, sim_kwargs):
    if name == 'GT2':
        return GT2(net, config_path='config/gt2.yaml', **sim_kwargs)
    elif name == 'LEACH':
        return LEACH(net, config_path='config/leach.yaml', **sim_kwargs)
    elif name == 'GTFR':
        return GTFR(net, config_path='config/gtfr.yaml', **sim_kwargs)
    elif name == 'DIA':
        return DIAMIA(net, config_path='config/dia_mia.yaml', **sim_kwargs)
    elif name == 'MIA':
        algo = DIAMIA(net, config_path='config/dia_mia.yaml', **sim_kwargs)
        algo.mode = 'MIA'
        return algo
    elif name == 'TCLE':
        return TCLE(net, config_path='config/tcle.yaml', **sim_kwargs)
    elif name == 'EFTCG-1':
        return EFTCG(net, config_path='config/eftcg.yaml', **sim_kwargs)
    elif name == 'EFTCG-2':
        algo = EFTCG(net, config_path='config/eftcg.yaml', **sim_kwargs)
        algo.k = 2
        return algo
    elif name == 'FL-LEACH-PSO':
        return FLLEACHPSO(net, config_path='config/fl_leach_pso.yaml', **sim_kwargs)
    elif name == 'SCA-LEVY':
        return SCALEVY(net, config_path='config/sca_levy.yaml', **sim_kwargs)
    elif name == 'FC-CRA':
        return FCCRA(net, config_path='config/fc_cra.yaml', **sim_kwargs)
    elif name == 'EE-TCM':
        return EETCM(net, config_path='config/ee_tcm.yaml', **sim_kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {name}')

########################################################################
# MAIN
########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSN topology-control benchmark')
    parser.add_argument('--algo', type=str, default=None, choices=ALGO_CHOICES,
                        help='Algorithm to run (default: GT2)')
    args = parser.parse_args()

    algorithm = args.algo or 'FC-CRA'

    net = build_network()
    print("Generated done")
    print("Possible connectivity:", net.check_potential_connectivity())

    sim_kwargs = dict(max_rounds=MAX_ROUNDS, plot_period=PLOT_PERIOD)
    algo = make_algo(algorithm, net, sim_kwargs)
    algo.run()

    print(f'\n=== RESULT [{algorithm}] ===')
    print(f'First node dead at round: {algo.t_no_dead}')
    print(f'Last node dead at round:  {algo.t}')
