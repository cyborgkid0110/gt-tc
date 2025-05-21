import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import qmc

num_nodes = 200
dead_nodes = 0
dead_nodes_t = []
area = 250
xs, ys = (0, 0)

e0 = 1                     # initial energy of the nodes
eth = 20                    # threshold energy
pth = 0.6*pow(10, -9)         # power threshold
hop_max = 2                 # number of neighbor hops
d = 50                      # cell size
Vsta = 3.6                  # standard working voltage
gamma = 5
p_strat = []
p_min = 0.01
p_max = 0.08
p_step = 0.0001
wave = 0.1224

e_mp = 0.0013*pow(10, -12)
e_fs = 10*pow(10, -12)
e_elec = 50*pow(10, -9)
e_agg = 5*pow(10, -9)
d0 = math.sqrt(e_fs/e_mp)
m_pkt_s = 20
m_pkt_l = 500
payoff = 0.02

def cal_rc(power):
    distance = math.sqrt((power*wave*wave)/(pth*16*math.pi*math.pi))
    # print(distance)
    return distance

def cal_tx_cost(d, role):
    m_bit = m_pkt_s if role == 'CM' else m_pkt_l
    c_tx = 0
    if d >= d0:
        c_tx = m_bit * e_elec + m_bit * e_mp * pow(d, 4)
    else:
        c_tx = m_bit * e_elec + m_bit * e_fs * pow(d, 2)
    return c_tx

test_node = None
def cal_cost(node_info, xn, yn, role):
    m_bit = 8
    d = math.sqrt(xn*xn + yn*yn)
    c_tx = cal_tx_cost(d, role)
    c_total = 0
    if role == 'CM':
        i_sense = random.uniform(pow(10, -8), 5*pow(10, -7))
        c_sense = node_info['Vpre'] * i_sense * m_bit
        i_process = random.uniform(pow(10, -8), 5*pow(10, -7))
        c_process = node_info['Vpre'] * m_bit * i_sense / 4
        c_total = c_sense + c_process + c_tx
        # Store costs for plotting
        sensing_costs.append(c_sense * 10000)
        processing_costs.append(c_process * 10000)
        transmitting_costs.append(c_tx * 10000)
    else:
        c_rx = m_pkt_s * e_elec
        c_agg = len(node_info['neighbors']) * m_pkt_s * e_agg
        c_total = c_rx + c_agg + c_tx
    return c_total

t = 0
max_t = 10000

node_dict = {}

network = {
    'edges': [],
    'vertices': [],
}

# Lists to store costs for plotting
sensing_costs = []
processing_costs = []
transmitting_costs = []
ch_costs = []
cm_costs = []

# Generate node with Poisson Disk Sampling
rng = np.random.default_rng()
radius = 30
engine = qmc.PoissonDisk(d=2, radius=radius, rng=rng, ncandidates=num_nodes, l_bounds=0, u_bounds=area * 2)
sample = engine.random(num_nodes)
not_generated_nodes = num_nodes - len(sample)
while (not_generated_nodes > 0):
    row = np.round(np.random.rand(1, 2) * area * 2, 2)
    sample = np.append(sample, row, axis=0)
    not_generated_nodes -= 1

for i in range(0, len(sample)):
    xn, yn = sample[i, 0] - area, sample[i, 1] - area
    network['vertices'].append((xn, yn))
    
    node_dict[(xn, yn)] = {
        'neighbors': [],
        'power': p_max / 4,
        'rc': cal_rc(p_max / 4),
        'e_res': e0,
        'Vpre': random.uniform(2.7, 4.2),
        'p0': None,
        'CH': False,
        'CH_neighbor': None,
        'c_ch': 0,
        'c_cm': 0,
    }

print("Generated done")

# Each node broadcast ADV message
# After receiving ADV, each node detect the number of neighboring nodes
for node1 in network['vertices']:
    for node2 in network['vertices']:
        if node1 == node2:
            continue
        d = math.hypot(node1[0] - node2[0], node1[1] - node2[1])
        if node_dict[node1]['rc'] >= d:
            node_dict[node1]['neighbors'].append(node2)

# ========== Start of Added Code for Data Collection ==========
# List to store dead_nodes across rounds
dead_nodes_values = []
# ========== End of Added Code for Data Collection ==========

while (t < max_t):
    CH_con = 0
    CH_can = 0
    CH_true = 0
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        if len(node_dict[node]['neighbors']) == 0:
            continue

        # If node has been selected as CH in the previous round, node can't join game in this round
        if node_dict[node]['CH'] == True:
            node_dict[node]['CH'] = False
            continue

        if test_node is None:
            test_node = node
        CH_con += 1
        c_ch = cal_cost(node_dict[node], node[0], node[1], "CH")
        c_cm = cal_cost(node_dict[node], node[0], node[1], "CM") 
        node_dict[node]['c_ch'] = c_ch
        node_dict[node]['c_cm'] = c_cm
        # Store CH and CM costs for plotting
        ch_costs.append(c_ch)
        cm_costs.append(c_cm)
        if c_ch-c_cm < 0:
            p0 = 0
        else:
            p0 = 1 - pow((c_ch-c_cm)/(payoff-c_cm), 1/(len(node_dict[node]['neighbors'])))
        node_dict[node]['p0'] = p0
        # if test_node == node:
        #     print(f"CH cost: {node_dict[node]['c_ch']}")
        #     print(f"CM cost: {node_dict[node]['c_cm']}")
        #     print(f"P0: {node_dict[node]['p0'] * 100}")
        #     print(f"Neighbors: {len(node_dict[node]['neighbors'])}")
        if random.random() < p0:
            p_ch = p0 * node_dict[node]['e_res'] / e0
            t_ge = p_ch / (1 - p_ch * (t % (1/p_ch)))
            if (random.uniform(0, 1) < t_ge):
                node_dict[node]['CH'] = True
                CH_true += 1      
            else:
                node_dict[node]['CH'] = False
            CH_can += 1

    # After real CHs are selected, CHs broadcast the advertising message
    # CMs receive all messages from CHs and select the nearest CH to send message.
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        if node_dict[node]['CH'] == True:
            node_dict[node]['e_res'] -= node_dict[node]['c_ch']
            for neighbor in node_dict[node]['neighbors']:
                if node_dict[neighbor]['CH_neighbor'] is None:
                    node_dict[neighbor]['CH_neighbor'] = node
                else:
                    xn, yn = neighbor
                    x_ch_old, y_ch_old = node_dict[neighbor]['CH_neighbor']
                    x_ch_temp, y_ch_temp = node
                    d1 = math.hypot(xn - x_ch_old, yn - y_ch_old)
                    d2 = math.hypot(xn - x_ch_temp, yn - y_ch_temp)
                    if d1 > d2:
                        node_dict[neighbor]['CH_neighbor'] = node
        else:
            node_dict[node]['e_res'] -= node_dict[node]['c_cm']
        
        # print(f'Node: {node}, residual energy: {node_dict[node]['e_res']}')
        if node_dict[node]['e_res'] <= 0:
            dead_nodes += 1
            dead_nodes_t.append(dead_nodes)

        if dead_nodes == num_nodes:
            break

    t += 1
    print(f'Round: {t}, Participated Nodes: {CH_con}, Total Candidate CH: {CH_can}, Total Real CH: {CH_true}')

    # ========== Start of Added Code for Data Collection Inside Loop ==========
    # Store dead_nodes for this round
    dead_nodes_values.append(dead_nodes)
    # ========== End of Added Code for Data Collection Inside Loop ==========

    # # Plotting node coverage
    # fig1, ax1 = plt.subplots(figsize=(6, 6))
    # ax1.set_xlim(-area, area)
    # ax1.set_ylim(-area, area)
    # ax1.set_aspect('equal', adjustable='box')
    # ax1.set_title("Sensor Node Coverage")
    # ax1.set_xlabel("X Position")
    # ax1.set_ylabel("Y Position")
    # ax1.grid(True)
    #
    # # Plot nodes and their coverage
    # for (x, y) in network['vertices']:
    #     circle = patches.Circle((x, y), node_dict[(x, y)]['rc'], edgecolor='blue', facecolor='lightblue', alpha=0.3)
    #     ax1.add_patch(circle)
    #     
    #     if node_dict[(x, y)]['CH'] == True:
    #         ax1.plot(x, y, 'ro')  # Red dot for CH
    #     else:
    #         ax1.plot(x, y, 'bo')  # Blue dot for non-CH
    #
    # node_patch = patches.Patch(color='lightblue', label='Coverage')
    # ax1.legend(handles=[node_patch])
    #
    # # Plotting sensing, processing, and transmitting costs as scatter plot
    # fig2, ax2 = plt.subplots(figsize=(10, 6))
    # node_indices = np.arange(len(sensing_costs))  # One point per node
    #
    # ax2.scatter(node_indices, sensing_costs, color='green', marker='o', label='Sensing Cost', alpha=0.6)
    # ax2.scatter(node_indices, processing_costs, color='blue', marker='^', label='Processing Cost', alpha=0.6)
    # ax2.scatter(node_indices, transmitting_costs, color='red', marker='s', label='Transmitting Cost', alpha=0.6)
    #
    # ax2.set_xlabel('Node Index')
    # ax2.set_ylabel('Cost (x 10^4)')
    # ax2.set_title('Sensing, Processing, and Transmitting Costs per Node')
    # ax2.legend()
    # ax2.grid(True)
    #
    # # Plotting CH and CM total costs as scatter plot
    # fig3, ax3 = plt.subplots(figsize=(10, 6))
    # node_indices_ch_cm = np.arange(len(ch_costs))  # One point per node
    #
    # ax3.scatter(node_indices_ch_cm, ch_costs, color='purple', marker='D', label='CH Total Cost', alpha=0.6)
    # ax3.scatter(node_indices_ch_cm, cm_costs, color='orange', marker='*', label='CM Total Cost', alpha=0.6)
    #
    # ax3.set_xlabel('Node Index')
    # ax3.set_ylabel('Total Cost')
    # ax3.set_title('Total Cost of Becoming CH vs CM per Node')
    # ax3.legend()
    # ax3.grid(True)
    #
    # # Display all plots
    # plt.show()

# ========== Start of Added Code for Plotting Dead Nodes Line Chart ==========
# Plotting dead_nodes as a line chart
fig4, ax4 = plt.subplots(figsize=(10, 6))
round_indices = np.arange(len(dead_nodes_values))

ax4.plot(round_indices, dead_nodes_values, color='red', marker='.', linestyle='-', label='Dead Nodes')
ax4.set_xlabel('Round')
ax4.set_ylabel('Number of Dead Nodes')
ax4.set_title('Number of Dead Nodes vs Round')
ax4.legend()
ax4.grid(True)

# Save the plot to a file
plt.savefig('dead_nodes_plot.png')
# Display the plot
plt.show()
# ========== End of Added Code for Plotting Dead Nodes Line Chart ==========