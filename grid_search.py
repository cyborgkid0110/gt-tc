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
import multiprocessing
import itertools
from datetime import datetime

num_nodes = 200
dead_nodes = 0
area = 250

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
# ALPHA = 1.5     # e_balance_benefit
# BETA = 1.5      # ctb_benefit
# M = 0.01        # e_cost

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
    else:
        c_rx = m_pkt_s * e_elec
        c_agg = len(node_info['neighbors']) * m_pkt_s * e_agg
        c_total = c_rx + c_agg + c_tx
    return c_total

def e_cost_func(x):
    return np.exp(x/10)

def cal_utility(node, power, hop_max=hop_max):
    utility = ctb_benefit(node, hop_max) - e_balance_benefit(node_dict[node]['local_net']) - e_cost(power, node_dict[node]['e_res'])
    return utility

def get_vertices(node, hop):
    vertices = set([node])

    if len(node_dict[node]['neighbors']) == 0 or node_dict[node]['CH'] == True:
        return list(vertices)

    if hop != 1:
        for neighbor in node_dict[node]['neighbors']:
            vertices.add(neighbor)
            sub_vertices = get_vertices(neighbor, hop - 1)
            vertices.update(sub_vertices)
    else:
        for neighbor_1_hop in node_dict[node]['neighbors']:
            vertices.add(neighbor_1_hop)

    return list(vertices)

def get_graph(node, hop):
    vertices = get_vertices(node, hop)
    edges = np.zeros((len(vertices), len(vertices)), dtype=int)

    for node1 in vertices:
        for node2 in vertices:
            if node1 == node2:
                continue

            if node2 in node_dict[node1]['neighbors']:
                i = vertices.index(node1)
                j = vertices.index(node2)
                edges[i, j] = 1

    return vertices, edges

def dfs(network, start_vertex, visited):
    i = network['vertices'].index(start_vertex)
    if visited[i] == True:
        return visited
    
    visited[i] = True
    for j in range(0, len(network['vertices'])):
        if network['edges'][i, j] == 1:
            next_vertex = network['vertices'][j]
            dfs(network, next_vertex, visited)

def check_connectivity(network, start_node):
    visited = [False] * len(network['vertices'])
    dfs(network, start_node, visited)

    if False in visited:
        return False

    return True

def update_global_network(global_network, local_network):
    length = len(local_network['vertices'])
    for i1 in range(0, length):
        for j1 in range(0, length):
            if i1 == j1:
                continue

            i2 = global_network['vertices'].index(local_network['vertices'][i1])
            j2 = global_network['vertices'].index(local_network['vertices'][j1])

            global_network['edges'][i2, j2] = local_network['edges'][i1, j1]

def main_alg(ALPHA, BETA, M):
    def e_cost(tx_power, e_res):
        T = 1.0
        cost = integrate.quad(e_cost_func, e0 - e_res, e0 - e_res + tx_power * T)
        return cost[0] / M

    def ctb_benefit(node, hop):
        vertices = get_vertices(node, hop)
        if len(vertices) == 0:
            return 0
        return BETA * len(vertices)

    def e_balance_benefit(network):
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

    t = 0
    max_t = 50000

    node_dict = {}

    network = {
        'edges': None,
        'vertices': [],
    }

    # Generate node with Poisson Disk Sampling
    seed = 42
    rng = np.random.default_rng(seed)
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

    # print("Generated done")

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
        
        # Each node broadcast ADV message
        # After receiving ADV, each node detect the number of neighboring nodes
        for i in range(0, num_nodes):
            node1 = network['vertices'][i]
            for j in range(0, num_nodes):
                if i == j:
                    continue
                node2 = network['vertices'][j]
                d = math.hypot(node1[0] - node2[0], node1[1] - node2[1])
                if node_dict[node1]['rc'] >= d:
                    network['edges'][i, j] = 1
                    add_neighbor(node1, node2)
                    

        for i in range(0, num_nodes):
            node = network['vertices'][i]
            if node_dict[node]['e_res'] <= 0 or len(node_dict[node]['neighbors']) == 0:
                continue

            # If node has been selected as CH in the previous round, node can't join game in this round
            # Reset CH node to default settings
            if node_dict[node]['CH'] == True:
                node_dict[node]['power'] = p_max / 4
                node_dict[node]['rc'] = cal_rc(p_max / 4)
                node_dict[node]['CH_neighbors'] = []
                node_dict[node]['CH'] = False
                continue

            CH_con += 1
            c_ch = cal_cost(node_dict[node], node[0], node[1], "CH")
            c_cm = cal_cost(node_dict[node], node[0], node[1], "CM") 
            node_dict[node]['c_ch'] = c_ch
            node_dict[node]['c_cm'] = c_cm
            if c_ch-c_cm < 0:
                p0 = 0
            else:
                p0 = 1 - pow((c_ch-c_cm)/(payoff-c_cm), 1/(len(node_dict[node]['neighbors'])))
            node_dict[node]['p0'] = p0
            if random.random() < p0:
                p_ch = p0 * node_dict[node]['e_res'] / e0
                node_dict[node]['p_ch'] = p_ch
                t_ge = p_ch / (1 - p_ch * (t % (1/p_ch)))
                # print(f'Node {i}: {t_ge}')
                if (random.uniform(0, 1) < p_ch):
                # if (random.uniform(0, 1) < t_ge):
                    node_dict[node]['CH'] = True
                    CH_true += 1      
                else:
                    node_dict[node]['CH'] = False
                CH_can += 1

        # After real CHs are selected, CHs broadcast the advertising message
        # CHs who receive advertising messages will add the node to CH neighboring list.
        # CMs receive all messages from CHs and select the nearest CH to send message.

        for i in range(0, num_nodes):
            ch_node = network['vertices'][i]     # ch_node means that only CHs are used in this loop
            if node_dict[ch_node]['e_res'] <= 0:
                continue
            if node_dict[ch_node]['CH'] == True: # CH set to max tx power
                node_dict[ch_node]['power'] = p_max
                node_dict[ch_node]['rc'] = cal_rc(p_max)
                node_dict[ch_node]['c_ch'] = cal_cost(node_dict[ch_node], ch_node[0], ch_node[1], "CH")
                
                for j in range(0, num_nodes):
                    if i == j:
                        continue
                    node = network['vertices'][j]
                    if node_dict[node]['e_res'] <= 0:
                        continue

                    d = math.hypot(ch_node[0] - node[0], ch_node[1] - node[1])
                    if node_dict[ch_node]['rc'] >= d: # node receives advertising message
                        network['edges'][i, j] = 1
                        # If received node is CH
                        if node_dict[node]['CH'] == True:
                            if node not in node_dict[ch_node]['CH_neighbors']:
                                node_dict[ch_node]['CH_neighbors'].append(node)
                        # If received node is CM, update the CH belong if new CH is closer to node.
                        else:
                            if node_dict[node]['CH_belong'] is None:
                                node_dict[node]['CH_belong'] = ch_node
                            else:
                                xn, yn = node
                                x_ch_old, y_ch_old = node_dict[node]['CH_belong']
                                x_ch_temp, y_ch_temp = ch_node
                                d1 = math.hypot(xn - x_ch_old, yn - y_ch_old)
                                d2 = math.hypot(xn - x_ch_temp, yn - y_ch_temp)
                                if d1 > d2:
                                    node_dict[node]['CH_belong'] = ch_node
        
        # Update neighbor list of nodes
        for i in range(0, num_nodes):
            node = network['vertices'][i]
            if node_dict[node]['e_res'] <= 0:
                continue

            # Node is CM
            if node_dict[node]['CH'] == False and node_dict[node]['CH_belong'] is not None:
                for neighbor in node_dict[node]['neighbors'][:]:
                    # Neighbor is CH
                    j = network['vertices'].index(neighbor)
                    if node_dict[neighbor]['CH'] == True:
                        if neighbor != node_dict[node]['CH_belong']:
                            network['edges'][i, j] = 0
                            delete_neighbor(node, neighbor)
                    else:
                        if node_dict[neighbor]['CH_belong'] != node_dict[node]['CH_belong']:
                            network['edges'][i, j] = 0
                            delete_neighbor(node, neighbor)
            # Node is CH
            elif node_dict[node]['CH'] == True:
                for neighbor in node_dict[node]['neighbors'][:]:
                    j = network['vertices'].index(neighbor)
                    if node_dict[node]['CH'] == True:
                        network['edges'][i, j] = 0
                        delete_neighbor(node, neighbor)
                    if node_dict[neighbor]['CH_belong'] != node:
                        network['edges'][i, j] = 0
                        delete_neighbor(node, neighbor)
            else:
                for neighbor in node_dict[node]['neighbors'][:]:
                    j = network['vertices'].index(neighbor)
                    network['edges'][i, j] = 0
                    delete_neighbor(node, neighbor)

        # Unconnected nodes will try to join a cluster by sending join request to surrounding node
        extended_cluster = 0
        while (extended_cluster == 0):
            extended_cluster = 1

            for i in range(0, num_nodes):
                # unjoined_node means that only nodes that not joined any cluster are used in this loop
                unjoined_node = network['vertices'][i]
                if node_dict[unjoined_node]['e_res'] < 0:
                    continue
                if node_dict[unjoined_node]['CH_belong'] is not None and node_dict[unjoined_node]['CH'] == False:
                    continue

                if node_dict[unjoined_node]['CH_belong'] is None and node_dict[unjoined_node]['CH'] == False:
                    # if there exists at least 1 node that haven't joined any cluster, flag is triggered
                    extended_cluster = 0

                final_cm_neighbor = None
                final_cm_neighbor_index = None
                neighbors_temp = []
                for j in range(0, num_nodes):
                    if i == j:
                        continue

                    cm_node = network['vertices'][j]
                    # since unjoined_node is impossible to reach the CHs, only need to focus on CMs.
                    if node_dict[cm_node]['CH_belong'] is None or node_dict[cm_node]['CH'] == True:
                        continue

                    d = math.hypot(unjoined_node[0] - cm_node[0], unjoined_node[1] - cm_node[1])
                    if node_dict[unjoined_node]['rc'] >= d:
                        neighbors_temp.append(cm_node)
                        if node_dict[unjoined_node]['CH_belong'] is None:
                            node_dict[unjoined_node]['CH_belong'] = node_dict[cm_node]['CH_belong']
                            final_cm_neighbor = cm_node
                            final_cm_neighbor_index = j
                        else:
                            xn, yn = unjoined_node
                            x_ch_old, y_ch_old = node_dict[unjoined_node]['CH_belong']    # old CH
                            x_ch_temp, y_ch_temp = node_dict[cm_node]['CH_belong']        # new CH
                            d1 = math.hypot(xn - x_ch_old, yn - y_ch_old)
                            d2 = math.hypot(xn - x_ch_temp, yn - y_ch_temp)
                            if d1 > d2:
                                node_dict[unjoined_node]['CH_belong'] = node_dict[cm_node]['CH_belong']
                                final_cm_neighbor = cm_node
                                final_cm_neighbor_index = j
                
                if final_cm_neighbor is not None:
                    network['edges'][final_cm_neighbor_index, i] = 1
                    add_neighbor(cm_node, unjoined_node)
                    # after removing neighbor not in same cluster, neighbors_temp is new neighbor list of unjoined_node
                    for neighbor in neighbors_temp[:]:
                        k = network['vertices'].index(neighbor)
                        if node_dict[neighbor]['CH_belong'] != node_dict[unjoined_node]['CH_belong']:
                            network['edges'][i, k] = 0
                            neighbors_temp.remove(neighbor)
                        else:
                            network['edges'][i, k] = 1
                            add_neighbor(unjoined_node, neighbor)
                    node_dict[unjoined_node]['neighbors'] = neighbors_temp
                else:
                    node_dict[unjoined_node]['power'] += p_step
                    node_dict[unjoined_node]['rc'] = cal_rc(node_dict[unjoined_node]['power'])

        modified_network = network
        cm_node_rc = cal_rc(p_max / 4)
        
        for i in range(0, num_nodes):
            ch_node = network['vertices'][i]
            if node_dict[ch_node]['e_res'] <= 0 or node_dict[ch_node]['CH'] == False:
                continue

            xn, yn = ch_node
            for neighbor in node_dict[ch_node]['neighbors']:
                if node_dict[neighbor]['CH'] == True:
                    continue

                x_neighbor, y_neighbor = neighbor
                d = math.hypot(xn - x_neighbor, yn - y_neighbor)
                if cm_node_rc < d:
                    j = network['vertices'].index(neighbor)
                    modified_network['edges'][i, j] = 0

            for neighbor in node_dict[ch_node]['CH_neighbors']:
                j = network['vertices'].index(neighbor)
                modified_network['edges'][i, j] = 0

        # directional_wsn_plot(modified_network, node_dict)   
        
        G = graph.build_graph(modified_network['vertices'], modified_network['edges'])
        layered_batches_per_cluster = graph.divide_network_by_clusters(G, node_dict)

        # Intra-cluster topology control:
        for ch_node, layers in layered_batches_per_cluster.items():         # layers of a cluster
            x_ch, y_ch = tuple(float(x) for x in ch_node)
            
            nash_eq = False
            while nash_eq == False:
                nash_eq = True
                for i, layer in reversed(list(enumerate(layers))):                          # each layer
                    if i == 0:
                        continue
                    for batch in layer:                                     # each batch in a layer
                        for node in batch:                                  # each node in a batch
                            cm_node = tuple(float(x) for x in node)
                            if node_dict[cm_node]['CH'] == True:
                                continue
                            if node_dict[cm_node]['local_net'] is None:
                                local_net = {'vertices': [], 'edges': []}
                                local_net['vertices'], local_net['edges'] = get_graph(cm_node, hop_max)
                                node_dict[cm_node]['local_net'] = local_net

                            e_res = node_dict[cm_node]['e_res']
                            if node_dict[cm_node]['util'] is None:
                                node_dict[cm_node]['util'] = cal_utility(cm_node, node_dict[cm_node]['power'])
                                # node_dict[cm_node]['util'] = ctb_benefit(cm_node, hop_max) - e_balance_benefit(node_dict[cm_node]['local_net']) - e_cost(node_dict[cm_node]['power'], e_res)
                            new_power = node_dict[cm_node]['power'] - p_step
                            if new_power < p_min:
                                new_power = p_min
                            new_rc = cal_rc(new_power)
                            new_util = None
                            topology_changed = False
                            old_neighbors = node_dict[cm_node]['neighbors']
                            new_local_net = node_dict[cm_node]['local_net']

                            # print("CH: ", ch_node)
                            # print("CM: ", cm_node)
                            # print(new_local_net)
                            # print(new_power)

                            # check connection between around each CM
                            for neighbor in node_dict[cm_node]['neighbors'][:]:
                                if cm_node == neighbor:
                                    continue
                                d = math.hypot(cm_node[0] - neighbor[0], cm_node[1] - neighbor[1])
                                if new_rc < d:     
                                    topology_changed = True
                                    # if cm_node not in new_local_net['vertices'] or neighbor not in new_local_net['vertices']:
                                    #     print("Neighbor", neighbor)
                                    #     print("CH_belong: ", node_dict[cm_node]['CH_belong'])
                                    #     directional_wsn_plot(modified_network, node_dict)
                                    j = new_local_net['vertices'].index(cm_node)
                                    k = new_local_net['vertices'].index(neighbor)
                                    new_local_net['edges'][j, k] = 0
                                    delete_neighbor(cm_node, neighbor)

                            new_local_net = {'vertices': [], 'edges': []}
                            new_local_net['vertices'], new_local_net['edges'] = get_graph(cm_node, hop_max)

                            connected = check_connectivity(new_local_net, cm_node)
                            if connected == False:
                                new_util = -100.0 * e_cost(new_power, e_res)
                            else:
                                new_util = cal_utility(cm_node, new_power)
                                # new_util = ctb_benefit(cm_node, hop_max) - e_balance_benefit(new_local_net) - e_cost(new_power, e_res)

                            if new_util > node_dict[cm_node]['util']:
                                node_dict[cm_node]['util'] = new_util
                                node_dict[cm_node]['power'] = new_power
                                node_dict[cm_node]['rc'] = new_rc
                                if topology_changed is True:
                                    node_dict[cm_node]['local_net'] = new_local_net
                                    update_global_network(modified_network, new_local_net)
                                nash_eq = False
                            else:
                                node_dict[cm_node]['neighbors'] = old_neighbors

        print(f'Iteration {t}: Finished, Candicate CH: {CH_can}, Real CH: {CH_true}')
        for i in range(0, num_nodes):
            ch_node = network['vertices'][i]
            if node_dict[ch_node]['e_res'] <= 0 or node_dict[ch_node]['CH'] == False:
                continue

            xn, yn = ch_node
            for neighbor in node_dict[ch_node]['neighbors']:
                if node_dict[neighbor]['CH'] == True:
                    continue

                x_neighbor, y_neighbor = neighbor
                d = math.hypot(xn - x_neighbor, yn - y_neighbor)
                if cm_node_rc < d:
                    j = network['vertices'].index(neighbor)
                    modified_network['edges'][i, j] = 0

            for neighbor in node_dict[ch_node]['CH_neighbors']:
                j = network['vertices'].index(neighbor)
                modified_network['edges'][i, j] = 0
                        
        # directional_wsn_plot(modified_network, node_dict)
        
        # maintainance phase
        for i in range(0, num_nodes):
            node = network['vertices'][i]
            
            if node_dict[node]['CH'] == True:
                node_dict[node]['e_res'] -= node_dict[node]['c_ch']
            else:
                node_dict[node]['e_res'] -= node_dict[node]['c_cm']
                
            if node_dict[node]['e_res'] < 0:
                node_dict[node]['e_res'] = 0
                end_loop = True
        
        if end_loop == True:
            break
        t += 1

    print(f'Iterations without dead node: {t}')
    return t

# Function to run main_alg with a parameter combination and return result
def evaluate_params(params):
    ALPHA, BETA, M = params
    try:
        t = main_alg(ALPHA, BETA, M)
        return (ALPHA, BETA, M, t)
    except Exception as e:
        return (ALPHA, BETA, M, None)

def grid_search_multiprocessing(alpha_range, beta_range, m_range, output_file):
    # Generate all parameter combinations
    param_combinations = list(itertools.product(alpha_range, beta_range, m_range))
    
    # Use multiprocessing Pool to parallelize evaluations
    num_processes = 5
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_params, param_combinations)
    
    # Save results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ALPHA', 'BETA', 'M', 'Score'])
        for result in results:
            writer.writerow(result)

if __name__ == '__main__':
    # Define large parameter ranges
    alpha_range = np.linspace(0.1, 1.0, 10)  # 100 values from 0.1 to 1.0
    beta_range = np.linspace(0.01, 0.1, 10)  # 100 values from 0.01 to 0.1
    m_range = np.arange(0.1, 10, 1)        # 100 values from 100 to 1000 (step ~9)
    
    # Output CSV file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'grid_search_results_{timestamp}.csv'
    
    # Run grid search
    results = grid_search_multiprocessing(alpha_range, beta_range, m_range, output_file)
    print(f"Results saved to {output_file}")
