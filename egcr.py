import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import qmc
from scipy import integrate
import csv

num_nodes = 200
dead_nodes = 0
dead_nodes_t = []
area = 250
xs, ys = (0, 0)

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
    
def cal_rx_power(p_tx, d):
    p_rx = (p_tx*wave*wave)/(16*math.pi*math.pi*d*d)
    return p_rx

def cal_tx_cost(d, role):
    m_bit = m_pkt_s if role == 'CM' else m_pkt_l
    c_tx = 0
    if d >= d0:
        c_tx = m_bit * e_elec + m_bit * e_mp * pow(d, 4)
    else:
        c_tx = m_bit * e_elec + m_bit * e_fs * pow(d, 2)
    return c_tx

def add_neighbor(node, neighbor):
    global node_dict

    if neighbor not in node_dict[node]['neighbors']:
        node_dict[node]['neighbors'].append(neighbor)

def delete_neighbor(node, neighbor):
    global node_dict

    if neighbor in node_dict[node]['neighbors']:
        node_dict[node]['neighbors'].remove(neighbor)

# def reset_node():
#     for node in network['vertices']:
#         node_dict[node][]

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

def e_cost_func(x):
    return np.exp(x/10)

def e_cost(tx_power, e_res):
    M = 0.01
    T = 1.0
    cost = integrate.quad(e_cost_func, e0 - e_res, e0 - e_res + tx_power * T)
    return cost[0] / M

def ctb_benefit(network):
    BETA = 1.5
    return BETA * len(network['vertices'])

def e_balance_benefit(network):
    ALPHA = 1.5
    e_res_all = []
    sum_e_res = 0
    avg_e_res = 0
    sum_diff = 0

    if len(network['vertices']) == 0:
        return 0
    
    for vertex in network['vertices']:
        sum_e_res += node_dict[vertex]['e_res']
        e_res_all.append(node_dict[vertex]['e_res'])
    avg_e_res = sum_e_res / len(network['vertices'])

    for e_res in e_res_all:
        sum_diff += (e_res - avg_e_res)*(e_res - avg_e_res)

    benefit = ALPHA * sum_diff / len(network['vertices'])
    return benefit

def get_graph(node, hop):
    vertices = set([node])
    edges = set()

    if len(node_dict[node]['neighbors']) == 0 or node_dict[node]['CH'] == True:
        return list(vertices), list(edges)

    if hop != 1:
        for neighbor in node_dict[node]['neighbors']:
            vertices.add(neighbor)
            edges.add((node, neighbor))
            sub_vertices, sub_edges = get_graph(neighbor, hop - 1)
            vertices.update(sub_vertices)
            edges.update(sub_edges)
    else:
        for neighbor_1_hop in node_dict[node]['neighbors']:
            vertices.add(neighbor_1_hop)
            edges.add((node, neighbor_1_hop))

    return list(vertices), list(edges)

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
        'CH_belong': None,
        'CH_neighbors': [],
        'c_ch': 0,
        'c_cm': 0,

        'util': None,
        'local_net': None
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
            add_neighbor(node1, node2)
            # node_dict[node1]['neighbors'].append(node2)

test_node = None
once = 0
while (t < max_t):
    CH_con = 0
    CH_can = 0
    CH_true = 0
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        # If node has been selected as CH in the previous round, node can't join game in this round
        # Reset CH node to default settings
        if node_dict[node]['CH'] == True:
            node_dict[node]['power'] = p_max / 4
            node_dict[node1]['rc'] = cal_rc(p_max / 4)
            node_dict[node]['CH_neighbors'] = []
            node_dict[node]['CH'] = False
            continue

        if len(node_dict[node]['neighbors']) == 0:
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
    # CHs who receive advertising messages will add the node to CH neighboring list.
    # CMs receive all messages from CHs and select the nearest CH to send message.
    for node1 in network['vertices']:
        if node_dict[node1]['e_res'] <= 0:
            continue
        if node_dict[node1]['CH'] == True: # CH set to max tx power
            node_dict[node1]['power'] = p_max
            node_dict[node1]['rc'] = cal_rc(p_max)
            node_dict[node1]['c_ch'] = cal_cost(node_dict[node], node[0], node[1], "CH")
            for node2 in network['vertices']:
                if node1 == node2:
                    continue
                if node_dict[node2]['e_res'] <= 0:
                    continue

                d = math.hypot(node1[0] - node2[0], node1[1] - node2[1])
                if node_dict[node1]['rc'] >= d: # node2 receives advertising message
                    # If received node is CH
                    if node_dict[node2]['CH'] == True:
                        if node2 not in node_dict[node1]['CH_neighbors']:
                            node_dict[node1]['CH_neighbors'].append(node2)
                    # If received node is CM, update the CH belong if new CH is closer to node.
                    else:
                        if node_dict[node2]['CH_belong'] is None:
                            node_dict[node2]['CH_belong'] = node1
                        else:
                            xn, yn = node2
                            x_ch_old, y_ch_old = node_dict[node2]['CH_belong']
                            x_ch_temp, y_ch_temp = node1
                            d1 = math.hypot(xn - x_ch_old, yn - y_ch_old)
                            d2 = math.hypot(xn - x_ch_temp, yn - y_ch_temp)
                            if d1 > d2:
                                node_dict[node2]['CH_belong'] = node1
    
    # Update neighbor list of nodes
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue
        
        if node_dict[node]['CH'] == False and node_dict[node]['CH_belong'] is not None:
            for neighbor in node_dict[node]['neighbors'][:]:
                if node_dict[neighbor]['CH'] == True:
                    if neighbor != node_dict[node]['CH_belong']:
                        node_dict[node]['neighbors'].remove(neighbor)
                else:      
                    if node_dict[neighbor]['CH_belong'] != node_dict[node]['CH_belong']:
                        node_dict[node]['neighbors'].remove(neighbor)
        else:
            for neighbor in node_dict[node]['neighbors'][:]:
                if node_dict[neighbor]['CH_belong'] != node:
                    node_dict[node]['neighbors'].remove(neighbor)
    
    # Unconnected nodes will try to join a cluster by sending join request to surrounding node
    extended_cluster = 0
    while (extended_cluster == 0):
        extended_cluster = 1
        for node in network['vertices']:
            if node_dict[node]['CH_belong'] is None and node_dict[node]['CH'] == False:
                extended_cluster = 0

                for node2 in network['vertices']:
                    if node == node2:
                        continue
                    d = math.hypot(node[0] - node2[0], node[1] - node2[1])
                    # Only CM can received the message. Node received first ACK will set CH belong
                    if node_dict[node]['rc'] >= d:
                        if node_dict[node2]['CH_belong'] is not None:
                            if node_dict[node]['power'] > node_dict[node2]['power']:
                                node_dict[node2]['power'] = node_dict[node]['power']
                                node_dict[node2]['rc'] = cal_rc(node_dict[node2]['power'])
                        
                            if node_dict[node]['CH_belong'] is None:
                                node_dict[node]['CH_belong'] = node_dict[node2]['CH_belong']

                            if node_dict[node]['CH_belong'] == node_dict[node2]['CH_belong']:
                                add_neighbor(node, node2)
                                add_neighbor(node2, node)

                # If no ACK is received, node increases the power transmission
                if node_dict[node]['CH_belong'] is None:
                    node_dict[node]['power'] += p_step
                    node_dict[node]['rc'] = cal_rc(node_dict[node]['power'])

    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        if node_dict[node]['CH'] == False and node_dict[node]['CH_belong'] is not None:
            add_neighbor(node_dict[node]['CH_belong'], node)

    # Check network
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        if node_dict[node]['CH'] == False:
            # for neighbor in node_dict[node]['CH_neighbors']:
            #     if (node, neighbor) not in network['edges'] and (neighbor, node) not in network['edges']:
            #         network['edges'].append((node, neighbor))
        # else:
            for neighbor in node_dict[node]['neighbors']:
                if (node, neighbor) not in network['edges'] and (neighbor, node) not in network['edges']:
                    network['edges'].append((node, neighbor))

    # Plotting node coverage
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_xlim(-area, area)
    ax2.set_ylim(-area, area)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title("Sensor Node Coverage")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(True)

    # Plot links between nodes
    for edge in network['edges']:
        node1, node2 = edge
        x_values = [node1[0], node2[0]]
        y_values = [node1[1], node2[1]]
        ax2.plot(x_values, y_values, 'k-', linewidth=1, alpha=0.5)  # Black lines for links

    # Plot nodes and their coverage
    for (x, y) in network['vertices']:
        if node_dict[(x, y)]['CH'] == True: 
            circle = patches.Circle((x, y), node_dict[(x, y)]['rc'], edgecolor='blue', facecolor='lightblue', alpha=0.3)
            ax2.add_patch(circle)
        
        if node_dict[(x, y)]['CH'] == True:
            ax2.plot(x, y, 'ro')  # Red dot for CH
        else:
            ax2.plot(x, y, 'bo')  # Blue dot for non-CH

    node_patch = patches.Patch(color='lightblue', label='Coverage')
    ax2.legend(handles=[node_patch])
    print(f'Round: {t}, Participated Nodes: {CH_con}, Total Candidate CH: {CH_can}, Total Real CH: {CH_true}')
    network['edges'] = []

    # Intra-cluster topology control:
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue
        if node_dict[node]['CH'] == True:
            test_node = None
            time = 0
            nash_eq = False
            while nash_eq is False:
                # print(f'Retry: {time}, Node: {node}')
                nash_eq = True
                for cm_node in node_dict[node]['neighbors']: # all CMs
                    if node_dict[cm_node]['local_net'] is None:
                        local_net = {'vertices': [], 'edges': []}
                        local_net['vertices'], local_net['edges'] = get_graph(cm_node, hop_max)
                        node_dict[cm_node]['local_net'] = local_net

                    e_res = node_dict[cm_node]['e_res']
                    if node_dict[cm_node]['util'] is None:
                        node_dict[cm_node]['util'] = ctb_benefit(node_dict[cm_node]['local_net']) - e_balance_benefit(node_dict[cm_node]['local_net']) - e_cost(node_dict[cm_node]['power'], e_res)
                    new_power = node_dict[cm_node]['power'] - p_step
                    if new_power < p_min:
                        new_power = p_min
                    new_util = None
                    topology_changed = False
                    old_neighbors = node_dict[cm_node]['neighbors']

                    # check connection between around each CM
                    for neighbor in old_neighbors:
                        d = math.hypot(cm_node[0] - neighbor[0], cm_node[1] - neighbor[1])
                        p_rx = cal_rx_power(new_power, d)
                        if p_rx < pth:      # remove connection if power is smaller than threshold
                            topology_changed = True
                            node_dict[cm_node]['neighbors'].remove(neighbor)

                    new_local_net = {'vertices': [], 'edges': []}
                    new_local_net['vertices'], new_local_net['edges'] = get_graph(cm_node, hop_max)

                    if len(new_local_net['vertices']) != len(local_net['vertices']):
                        new_util = -100.0 * e_cost(new_power, e_res)
                    else:
                        # print('-------------------------------------------')
                        # print('Node: ', cm_node)
                        # print('Contribute benefit: ', ctb_benefit(new_local_net))
                        # print('Energy balance benefit: ', e_balance_benefit(new_local_net))
                        # print('Cost: ', e_cost(new_power, e_res))
                        new_util = ctb_benefit(new_local_net) - e_balance_benefit(new_local_net) - e_cost(new_power, e_res)

                    if new_util > node_dict[cm_node]['util']:
                        node_dict[cm_node]['util'] = new_util
                        node_dict[cm_node]['power'] = new_power
                        nash_eq = False
                        if len(node_dict[cm_node]['neighbors']) != len(old_neighbors):
                            print("Change neighbor", cm_node)
                        if test_node == None:
                            test_node = cm_node

                        if cm_node == test_node and once < 200:
                            once += 1
                            # print('-------------------------------------------')
                            # print(f'Node: {cm_node}, new power: {new_power}')
                            # print('Contribute benefit: ', ctb_benefit(new_local_net))
                            # print('Energy balance benefit: ', e_balance_benefit(new_local_net))
                            # print('Cost: ', e_cost(new_power, e_res))
                        if topology_changed is True:
                            node_dict[cm_node]['local_net'] = new_local_net
                    else:
                        node_dict[cm_node]['neighbors'] = old_neighbors

                if nash_eq is False:
                    time += 1

    # Final achieved network
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        if node_dict[node]['CH'] == False:
            # for neighbor in node_dict[node]['CH_neighbors']:
            #     if (node, neighbor) not in network['edges'] and (neighbor, node) not in network['edges']:
            #         network['edges'].append((node, neighbor))
        # else:
            for neighbor in node_dict[node]['neighbors']:
                if (node, neighbor) not in network['edges'] and (neighbor, node) not in network['edges']:
                    network['edges'].append((node, neighbor))

    # Enengy consumption
    for node in network['vertices']:
        if node_dict[node]['e_res'] <= 0:
            continue

        if node_dict[node]['CH'] == True:
            node_dict[node]['e_res'] -= node_dict[node]['c_ch']
        else:
            node_dict[node]['e_res'] -= node_dict[node]['c_cm']

        # print(f'Node: {node}, residual energy: {node_dict[node]['e_res']}')
        if node_dict[node]['e_res'] <= 0:
            dead_nodes += 1

    dead_nodes_t.append(dead_nodes)
    if dead_nodes == num_nodes:
        break

    t += 1
    # print(f'Round: {t}, Participated Nodes: {CH_con}, Total Candidate CH: {CH_can}, Total Real CH: {CH_true}')

    # Plotting node coverage
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_xlim(-area, area)
    ax1.set_ylim(-area, area)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("Sensor Node Coverage")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid(True)

    # Plot links between nodes
    for edge in network['edges']:
        node1, node2 = edge
        x_values = [node1[0], node2[0]]
        y_values = [node1[1], node2[1]]
        ax1.plot(x_values, y_values, 'k-', linewidth=1, alpha=0.5)  # Black lines for links

    # Plot nodes and their coverage
    for (x, y) in network['vertices']:
        if node_dict[(x, y)]['CH'] == True: 
            circle = patches.Circle((x, y), node_dict[(x, y)]['rc'], edgecolor='blue', facecolor='lightblue', alpha=0.3)
            ax1.add_patch(circle)
        
        if node_dict[(x, y)]['CH'] == True:
            ax1.plot(x, y, 'ro')  # Red dot for CH
        else:
            ax1.plot(x, y, 'bo')  # Blue dot for non-CH

    node_patch = patches.Patch(color='lightblue', label='Coverage')
    ax1.legend(handles=[node_patch])

    # # Plotting sensing, processing, and transmitting costs as scatter plot
    # fig2, ax2 = plt.subplots(figsize=(10, 6))
    # node_indices = np.arange(len(sensing_costs))  # One point per node

    # ax2.scatter(node_indices, sensing_costs, color='green', marker='o', label='Sensing Cost', alpha=0.6)
    # ax2.scatter(node_indices, processing_costs, color='blue', marker='^', label='Processing Cost', alpha=0.6)
    # ax2.scatter(node_indices, transmitting_costs, color='red', marker='s', label='Transmitting Cost', alpha=0.6)

    # ax2.set_xlabel('Node Index')
    # ax2.set_ylabel('Cost (x 10^4)')
    # ax2.set_title('Sensing, Processing, and Transmitting Costs per Node')
    # ax2.legend()
    # ax2.grid(True)

    # # Plotting CH and CM total costs as scatter plot
    # fig3, ax3 = plt.subplots(figsize=(10, 6))
    # node_indices_ch_cm = np.arange(len(ch_costs))  # One point per node

    # ax3.scatter(node_indices_ch_cm, ch_costs, color='purple', marker='D', label='CH Total Cost', alpha=0.6)
    # ax3.scatter(node_indices_ch_cm, cm_costs, color='orange', marker='*', label='CM Total Cost', alpha=0.6)

    # ax3.set_xlabel('Node Index')
    # ax3.set_ylabel('Total Cost')
    # ax3.set_title('Total Cost of Becoming CH vs CM per Node')
    # ax3.legend()
    # ax3.grid(True)

    # Display all plots
    plt.show()