import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cupy as cp
import numpy as np
from numba import jit, cuda, float64, int32
from scipy.stats import qmc
from plot import directional_wsn_plot
import graph

num_nodes = 200
dead_nodes = 0
dead_nodes_t = []
area = 250
xs, ys = (0, 0)

# System parameters
e0 = 1
eth = 20
pth = 0.5 * pow(10, -9)
hop_max = 2
d = 50
Vsta = 3.6
gamma = 5
p_strat = []
p_min = 0.01
p_max = 0.08
p_step = 0.0001
wave = 0.1224

# Game 1 parameters
e_mp = 0.0013 * pow(10, -12)
e_fs = 10 * pow(10, -12)
e_elec = 50 * pow(10, -9)
e_agg = 5 * pow(10, -9)
d0 = math.sqrt(e_fs / e_mp)
m_pkt_s = 20
m_pkt_l = 500
payoff = 0.02

# Game 2 parameters
ALPHA = 1.5
BETA = 1.5
M = 0.01

# Initialize random seed for CuPy and Numba
cp.random.seed(42)

@jit(nopython=True)
def cal_rc(power):
    distance = math.sqrt((power * wave * wave) / (pth * 16 * math.pi * math.pi))
    return distance

@jit(nopython=True)
def cal_rx_power(p_tx, d):
    p_rx = (p_tx * wave * wave) / (16 * math.pi * math.pi * d * d)
    return p_rx

@jit(nopython=True)
def cal_tx_cost(d, role):
    m_bit = m_pkt_s if role == 'CM' else m_pkt_l
    c_tx = 0
    if d >= d0:
        c_tx = m_bit * e_elec + m_bit * e_mp * (d ** 4)
    else:
        c_tx = m_bit * e_elec + m_bit * e_fs * (d ** 2)
    return c_tx

@jit(nopython=True)
def cal_cost(xn, yn, Vpre, neighbors_len, role):
    m_bit = 8
    d = math.sqrt(xn * xn + yn * yn)
    c_tx = cal_tx_cost(d, role)
    c_total = 0
    if role == 'CM':
        i_sense = np.random.uniform(1e-8, 5e-7)
        c_sense = Vpre * i_sense * m_bit
        i_process = np.random.uniform(1e-8, 5e-7)
        c_process = Vpre * m_bit * i_sense / 4
        c_total = c_sense + c_process + c_tx
        return c_total, c_sense * 10000, c_process * 10000, c_tx * 10000
    else:
        c_rx = m_pkt_s * e_elec
        c_agg = neighbors_len * m_pkt_s * e_agg
        c_total = c_rx + c_agg + c_tx
        return c_total, 0.0, 0.0, c_tx * 10000

def add_neighbor(node, neighbor, node_dict):
    if neighbor not in node_dict[node]['neighbors']:
        node_dict[node]['neighbors'].append(neighbor)

def delete_neighbor(node, neighbor, node_dict):
    if neighbor in node_dict[node]['neighbors']:
        node_dict[node]['neighbors'].remove(neighbor)

def e_cost_func(x):
    return cp.exp(x / 10)

def e_cost(tx_power, e_res):
    T = 1.0
    a = e0 - e_res
    b = e0 - e_res + tx_power * T
    n = 1000
    x = cp.linspace(a, b, n)
    y = e_cost_func(x)
    dx = (b - a) / (n - 1)
    cost = cp.trapz(y, dx=dx)
    return cost / M

@jit(nopython=True)
def ctb_benefit(vertices_len):
    if vertices_len == 0:
        return 0.0
    return BETA * vertices_len

@jit(nopython=True)
def e_balance_benefit(e_res_array, vertices_len):
    if vertices_len == 0:
        return 0.0
    sum_e_res = 0.0
    for e_res in e_res_array:
        sum_e_res += e_res
    avg_e_res = sum_e_res / vertices_len
    sum_diff = 0.0
    for e_res in e_res_array:
        sum_diff += (e_res - avg_e_res) * (e_res - avg_e_res)
    benefit = ALPHA * sum_diff / vertices_len
    return benefit

@jit(nopython=True)
def cal_utility(ctb, e_balance, e_cost_val):
    return ctb - e_balance - e_cost_val

def get_vertices(node, hop, node_dict, all_vertices):
    vertices = []
    visited = set()

    @jit(nopython=True)
    def collect_vertices(node_idx, hop, node_coords, neighbor_list, vertex_list, visited_array):
        if visited_array[node_idx]:
            return
        visited_array[node_idx] = True
        vertex_list.append(node_idx)
        if hop == 0 or len(neighbor_list[node_idx]) == 0:
            return
        for neighbor_idx in neighbor_list[node_idx]:
            if not visited_array[neighbor_idx]:
                collect_vertices(neighbor_idx, hop - 1, node_coords, neighbor_list, vertex_list, visited_array)

    # Prepare data for Numba
    node_coords = [(n[0], n[1]) for n in all_vertices]
    node_indices = {n: i for i, n in enumerate(all_vertices)}
    neighbor_list = [node_dict[n]['neighbors'] for n in all_vertices]
    vertex_list = []
    visited_array = np.zeros(len(all_vertices), dtype=np.bool_)
    node_idx = node_indices[node]
    if not node_dict[node]['CH']:
        collect_vertices(node_idx, hop, np.array(node_coords, dtype=np.float64), neighbor_list, vertex_list, visited_array)
    else:
        vertex_list.append(node_idx)
    vertices = [all_vertices[i] for i in vertex_list]
    return vertices

def get_graph(node, hop, node_dict):
    vertices = get_vertices(node, hop, node_dict, network['vertices'])
    if not vertices:  # Handle empty vertices case
        return [], cp.zeros((0, 0), dtype=cp.int32)
    edges = cp.zeros((len(vertices), len(vertices)), dtype=cp.int32)
    for node1 in vertices:
        for node2 in vertices:
            if node1 == node2:
                continue
            if node2 in node_dict[node1]['neighbors']:
                i = vertices.index(node1)
                j = vertices.index(node2)
                edges[i, j] = 1
    return vertices, edges

@jit(nopython=True)
def dfs(edges, vertices_len, start_idx, visited):
    visited[start_idx] = True
    for j in range(vertices_len):
        if edges[start_idx, j] == 1 and not visited[j]:
            dfs(edges, vertices_len, j, visited)

def check_connectivity(network, start_node):
    vertices_len = len(network['vertices'])
    if vertices_len == 0:
        return True
    visited = np.zeros(vertices_len, dtype=np.bool_)
    start_idx = network['vertices'].index(start_node)
    dfs(cp.asnumpy(network['edges']), vertices_len, start_idx, visited)
    return np.all(visited)

@jit(nopython=True)
def update_global_network(global_edges, global_vertices, local_edges, local_vertices):
    length = len(local_vertices)
    for i1 in range(length):
        for j1 in range(length):
            if i1 == j1:
                continue
            i2 = global_vertices.index(local_vertices[i1])
            j2 = global_vertices.index(local_vertices[j1])
            global_edges[i2, j2] = local_edges[i1, j1]

# CUDA kernel for distance computation
@cuda.jit
def compute_distances(nodes, rc, edges):
    i = cuda.grid(1)
    if i >= nodes.shape[0]:
        return
    for j in range(nodes.shape[0]):
        if i == j:
            continue
        d = math.sqrt((nodes[i, 0] - nodes[j, 0]) ** 2 + (nodes[i, 1] - nodes[j, 1]) ** 2)
        if d <= rc[i]:
            edges[i, j] = 1

# Generate nodes with Poisson Disk Sampling (CPU-based)
rng = np.random.default_rng(42)
radius = 30
engine = qmc.PoissonDisk(d=2, radius=radius, rng=rng, ncandidates=num_nodes, l_bounds=0, u_bounds=area * 2)
sample = engine.random(num_nodes)
not_generated_nodes = num_nodes - len(sample)

while not_generated_nodes > 0:
    row = np.round(np.random.uniform(0, area * 2, size=(1, 2)), 2)
    sample = np.append(sample, row, axis=0)
    not_generated_nodes -= 1

# Transfer to GPU
sample = cp.array(sample)

node_dict = {}
network = {
    'edges': None,
    'vertices': [],
}

# Lists to store costs for plotting
sensing_costs = []
processing_costs = []
transmitting_costs = []
ch_costs = []
cm_costs = []

for i in range(len(sample)):
    xn, yn = float(sample[i, 0] - area), float(sample[i, 1] - area)
    network['vertices'].append((xn, yn))
    node_dict[(xn, yn)] = {
        'id': i,
        'neighbors': [],
        'power': p_max / 4,
        'rc': cal_rc(p_max / 4),
        'e_res': e0,
        'Vpre': float(np.random.uniform(2.7, 4.2)),
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

# Optimize CUDA grid size
block_size = 256
grid_size = (num_nodes + block_size - 1) // block_size

t = 0
max_t = 50000

while t < max_t:
    CH_con = 0
    CH_can = 0
    CH_true = 0
    end_loop = False

    # Reset network
    network['edges'] = cp.zeros((num_nodes, num_nodes), dtype=cp.int32)
    rc_array = cp.array([node_dict[node]['rc'] for node in network['vertices']])
    compute_distances[grid_size, block_size](sample, rc_array, network['edges'])
    for i in range(num_nodes):
        node = network['vertices'][i]
        node_dict[node]['power'] = p_max / 4
        node_dict[node]['rc'] = cal_rc(p_max / 4)
        node_dict[node]['neighbors'] = []
        node_dict[node]['CH_belong'] = None
        node_dict[node]['util'] = None
        node_dict[node]['local_net'] = None
        for j in range(num_nodes):
            if i == j:
                continue
            if network['edges'][i, j] == 1:
                add_neighbor(node, network['vertices'][j], node_dict)

    for i in range(num_nodes):
        node = network['vertices'][i]
        if node_dict[node]['e_res'] <= 0 or len(node_dict[node]['neighbors']) == 0:
            continue
        if node_dict[node]['CH']:
            node_dict[node]['power'] = p_max / 4
            node_dict[node]['rc'] = cal_rc(p_max / 4)
            node_dict[node]['CH_neighbors'] = []
            node_dict[node]['CH'] = False
            continue
        CH_con += 1
        c_ch, c_sense_ch, c_process_ch, c_tx_ch = cal_cost(node[0], node[1], node_dict[node]['Vpre'], len(node_dict[node]['neighbors']), "CH")
        c_cm, c_sense_cm, c_process_cm, c_tx_cm = cal_cost(node[0], node[1], node_dict[node]['Vpre'], len(node_dict[node]['neighbors']), "CM")
        node_dict[node]['c_ch'] = c_ch
        node_dict[node]['c_cm'] = c_cm
        ch_costs.append(c_ch)
        cm_costs.append(c_cm)
        sensing_costs.append(c_sense_cm)
        processing_costs.append(c_process_cm)
        transmitting_costs.append(c_tx_cm)
        if c_ch - c_cm < 0:
            p0 = 0
        else:
            p0 = 1 - (c_ch - c_cm) / (payoff - c_cm) ** (1 / len(node_dict[node]['neighbors']))
        node_dict[node]['p0'] = p0
        if np.random.uniform(0, 1) < p0:
            p_ch = p0 * node_dict[node]['e_res'] / e0
            node_dict[node]['p_ch'] = p_ch
            if np.random.uniform(0, 1) < p_ch:
                node_dict[node]['CH'] = True
                CH_true += 1
            else:
                node_dict[node]['CH'] = False
            CH_can += 1

    for i in range(num_nodes):
        ch_node = network['vertices'][i]
        if node_dict[ch_node]['e_res'] <= 0:
            continue
        if node_dict[ch_node]['CH']:
            node_dict[ch_node]['power'] = p_max
            node_dict[ch_node]['rc'] = cal_rc(p_max)
            node_dict[ch_node]['c_ch'] = cal_cost(ch_node[0], ch_node[1], node_dict[ch_node]['Vpre'], len(node_dict[ch_node]['neighbors']), "CH")[0]
            rc_array[i] = node_dict[ch_node]['rc']
            network['edges'][i, :] = 0
            compute_distances[grid_size, block_size](sample, rc_array, network['edges'])
            for j in range(num_nodes):
                if i == j:
                    continue
                node = network['vertices'][j]
                if node_dict[node]['e_res'] <= 0:
                    continue
                if network['edges'][i, j] == 1:
                    if node_dict[node]['CH']:
                        if node not in node_dict[ch_node]['CH_neighbors']:
                            node_dict[ch_node]['CH_neighbors'].append(node)
                    else:
                        if node_dict[node]['CH_belong'] is None:
                            node_dict[ch_node]['neighbors'].append(node)
                            node_dict[node]['CH_belong'] = ch_node
                        else:
                            xn, yn = node
                            x_ch_old, y_ch_old = node_dict[node]['CH_belong']
                            x_ch_temp, y_ch_temp = ch_node
                            d1 = math.hypot(xn - x_ch_old, yn - y_ch_old)
                            d2 = math.hypot(xn - x_ch_temp, yn - y_ch_temp)
                            if d1 > d2:
                                node_dict[node]['CH_belong'] = ch_node
                                node_dict[ch_node]['neighbors'].append(node)

    for i in range(num_nodes):
        node = network['vertices'][i]
        if node_dict[node]['e_res'] <= 0:
            continue
        if node_dict[node]['CH'] == False and node_dict[node]['CH_belong'] is not None:
            for neighbor in node_dict[node]['neighbors'][:]:
                j = network['vertices'].index(neighbor)
                if node_dict[neighbor]['CH']:
                    if neighbor != node_dict[node]['CH_belong']:
                        network['edges'][i, j] = 0
                        delete_neighbor(node, neighbor, node_dict)
                else:
                    if node_dict[neighbor]['CH_belong'] != node_dict[node]['CH_belong']:
                        network['edges'][i, j] = 0
                        delete_neighbor(node, neighbor, node_dict)
        elif node_dict[node]['CH']:
            for neighbor in node_dict[node]['neighbors'][:]:
                j = network['vertices'].index(neighbor)
                if node_dict[neighbor]['CH_belong'] != node:
                    network['edges'][i, j] = 0
                    delete_neighbor(node, neighbor, node_dict)
        else:
            for neighbor in node_dict[node]['neighbors'][:]:
                j = network['vertices'].index(neighbor)
                network['edges'][i, j] = 0
                delete_neighbor(node, neighbor, node_dict)

    extended_cluster = 0
    while extended_cluster == 0:
        extended_cluster = 1
        for i in range(num_nodes):
            unjoined_node = network['vertices'][i]
            if node_dict[unjoined_node]['e_res'] < 0:
                continue
            if node_dict[unjoined_node]['CH_belong'] is not None and node_dict[unjoined_node]['CH'] == False:
                continue
            if node_dict[unjoined_node]['CH_belong'] is None and node_dict[unjoined_node]['CH'] == False:
                extended_cluster = 0
            final_cm_neighbor = None
            final_cm_neighbor_index = None
            neighbors_temp = []
            for j in range(num_nodes):
                if i == j:
                    continue
                cm_node = network['vertices'][j]
                if node_dict[cm_node]['CH_belong'] is None or node_dict[cm_node]['CH']:
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
                        x_ch_old, y_ch_old = node_dict[unjoined_node]['CH_belong']
                        x_ch_temp, y_ch_temp = node_dict[cm_node]['CH_belong']
                        d1 = math.hypot(xn - x_ch_old, yn - y_ch_old)
                        d2 = math.hypot(xn - x_ch_temp, yn - y_ch_temp)
                        if d1 > d2:
                            node_dict[unjoined_node]['CH_belong'] = node_dict[cm_node]['CH_belong']
                            final_cm_neighbor = cm_node
                            final_cm_neighbor_index = j
            if final_cm_neighbor is not None:
                network['edges'][final_cm_neighbor_index, i] = 1
                add_neighbor(cm_node, unjoined_node, node_dict)
                for neighbor in neighbors_temp[:]:
                    k = network['vertices'].index(neighbor)
                    if node_dict[neighbor]['CH_belong'] != node_dict[unjoined_node]['CH_belong']:
                        network['edges'][i, k] = 0
                        neighbors_temp.remove(neighbor)
                    else:
                        network['edges'][i, k] = 1
                        add_neighbor(unjoined_node, neighbor, node_dict)
                node_dict[unjoined_node]['neighbors'] = neighbors_temp
            else:
                node_dict[unjoined_node]['power'] += p_step
                node_dict[unjoined_node]['rc'] = cal_rc(node_dict[unjoined_node]['power'])

    modified_network = network
    cm_node_rc = cal_rc(p_max / 4)
    for i in range(num_nodes):
        ch_node = network['vertices'][i]
        if node_dict[ch_node]['e_res'] <= 0 or node_dict[ch_node]['CH'] == False:
            continue
        xn, yn = ch_node
        for neighbor in node_dict[ch_node]['neighbors']:
            if node_dict[neighbor]['CH']:
                continue
            x_neighbor, y_neighbor = neighbor
            d = math.hypot(xn - x_neighbor, yn - y_neighbor)
            if cm_node_rc < d:
                j = network['vertices'].index(neighbor)
                modified_network['edges'][i, j] = 0
        for neighbor in node_dict[ch_node]['CH_neighbors']:
            j = network['vertices'].index(neighbor)
            modified_network['edges'][i, j] = 0

    modified_network_np = {
        'vertices': modified_network['vertices'],
        'edges': cp.asnumpy(modified_network['edges'])
    }
    # directional_wsn_plot(modified_network_np, node_dict)

    G = graph.build_graph(modified_network['vertices'], cp.asnumpy(modified_network['edges']))
    layered_batches_per_cluster = graph.divide_network_by_clusters(G, node_dict)

    for ch_node, layers in layered_batches_per_cluster.items():
        x_ch, y_ch = tuple(float(x) for x in ch_node)
        nash_eq = False
        while not nash_eq:
            nash_eq = True
            for i, layer in reversed(list(enumerate(layers))):
                if i == 0:
                    continue
                for batch in layer:
                    for node in batch:
                        cm_node = tuple(float(x) for x in node)
                        if node_dict[cm_node]['CH']:
                            continue
                        if node_dict[cm_node]['local_net'] is None:
                            local_net = {'vertices': [], 'edges': []}
                            local_net['vertices'], local_net['edges'] = get_graph(cm_node, hop_max, node_dict)
                            node_dict[cm_node]['local_net'] = local_net
                        e_res = node_dict[cm_node]['e_res']
                        vertices = get_vertices(cm_node, hop_max, node_dict, network['vertices'])
                        vertices_len = len(vertices)
                        e_res_array = np.array([node_dict[v]['e_res'] for v in vertices], dtype=np.float64)
                        ctb = ctb_benefit(vertices_len)
                        e_balance = e_balance_benefit(e_res_array, vertices_len)
                        if node_dict[cm_node]['util'] is None:
                            e_cost_val = e_cost(node_dict[cm_node]['power'], e_res)
                            node_dict[cm_node]['util'] = cal_utility(ctb, e_balance, e_cost_val)
                        new_power = node_dict[cm_node]['power'] - p_step
                        if new_power < p_min:
                            new_power = p_min
                        new_rc = cal_rc(new_power)
                        new_util = None
                        topology_changed = False
                        old_neighbors = node_dict[cm_node]['neighbors']
                        new_local_net = node_dict[cm_node]['local_net']
                        for neighbor in node_dict[cm_node]['neighbors'][:]:
                            if cm_node == neighbor:
                                continue
                            d = math.hypot(cm_node[0] - neighbor[0], cm_node[1] - neighbor[1])
                            if new_rc < d:
                                topology_changed = True
                                j = new_local_net['vertices'].index(cm_node)
                                k = new_local_net['vertices'].index(neighbor)
                                new_local_net['edges'][j, k] = 0
                                delete_neighbor(cm_node, neighbor, node_dict)
                        new_local_net = {'vertices': [], 'edges': []}
                        new_local_net['vertices'], new_local_net['edges'] = get_graph(cm_node, hop_max, node_dict)
                        connected = check_connectivity(new_local_net, cm_node)
                        if not connected:
                            new_util = -100.0 * e_cost(new_power, e_res)
                        else:
                            new_e_cost = e_cost(new_power, e_res)
                            new_util = cal_utility(ctb, e_balance, new_e_cost)
                        if new_util > node_dict[cm_node]['util']:
                            node_dict[cm_node]['util'] = new_util
                            node_dict[cm_node]['power'] = new_power
                            node_dict[cm_node]['rc'] = new_rc
                            if topology_changed:
                                node_dict[cm_node]['local_net'] = new_local_net
                                update_global_network(cp.asnumpy(modified_network['edges']), modified_network['vertices'], cp.asnumpy(new_local_net['edges']), new_local_net['vertices'])
                                modified_network['edges'] = cp.array(modified_network['edges'])
                            nash_eq = False
                        else:
                            node_dict[cm_node]['neighbors'] = old_neighbors

    print(f'Iteration {t}: Finished, Candidate CH: {CH_can}, Real CH: {CH_true}')
    for i in range(num_nodes):
        ch_node = network['vertices'][i]
        if node_dict[ch_node]['e_res'] <= 0 or node_dict[ch_node]['CH'] == False:
            continue
        xn, yn = ch_node
        for neighbor in node_dict[ch_node]['neighbors']:
            if node_dict[neighbor]['CH']:
                continue
            x_neighbor, y_neighbor = neighbor
            d = math.hypot(xn - x_neighbor, yn - y_neighbor)
            if cm_node_rc < d:
                j = network['vertices'].index(neighbor)
                modified_network['edges'][i, j] = 0
        for neighbor in node_dict[ch_node]['CH_neighbors']:
            j = network['vertices'].index(neighbor)
            modified_network['edges'][i, j] = 0

    for i in range(num_nodes):
        node = network['vertices'][i]
        if node_dict[node]['CH']:
            node_dict[node]['e_res'] -= node_dict[node]['c_ch']
        else:
            node_dict[node]['e_res'] -= node_dict[node]['c_cm']
        if node_dict[node]['e_res'] < 0:
            node_dict[node]['e_res'] = 0
            end_loop = True

    if end_loop:
        break
    t += 1

print(f'Iterations without dead node: {t}')