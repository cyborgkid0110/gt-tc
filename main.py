import random
import matplotlib.pyplot as plt
import plot
import math
from scipy import integrate
import numpy as np

# # Set random seed for reproducibility (optional)
# random.seed()

# Create list to store nodes
nodes = []
num_nodes = 200
area = 250
xs, ys = (0, 0) # sink node

e0 = 50                     # initial energy of the nodes
eth = 20                    # threshold energy
pth = 5*pow(10, -9)         # power threshold
hop_max = 2                 # number of neighbor hops
d = 50                      # cell size
Vsta = 3.6                  # standard working voltage
gamma = 5
p_strat = []
p_min = 0.01
p_max = 0.08
p_step = 0.0001
wave = 0.1224
# time
t = 0
max_t = 500

cells = []
cell_group = {}

# cells = [
#     (0, 0),...
# ]

# cell_group = {
#     (0, 0): {
#         'nodes': [{
#               'pos': (0, 0),
#               'energy': 0,
#         },...],
#         'active': None,
#         'energy': 0,
#         'idle_time': 0,
#         'power': 0, 
#         'util': 0, 
#         'neighbors': [{
#             'cell': (0,0),
#             'node': (0,0),
#         }, ...],
#         'local_net': {
#             'nodes': [(0,0), ...],
#             'links': [(0,0), ...],
#         }
#     },
#     ...
# }

# Generate 50 nodes
for i in range(num_nodes):
    x = random.uniform(-area, area)  # Random x coordinate between -100 and 100
    y = random.uniform(-area, area)  # Random y coordinate between -100 and 100
    nodes.append((round(x, 1), round(y, 1)))

def mapping_to_cell(node):
    xn, yn = node
    if xn > xs:
        xc = math.ceil((xn - xs)/d)
    else:
        xc = math.floor((xn - xs)/d)

    if yn > ys:
        yc = math.ceil((yn - ys)/d)
    else:
        yc = math.floor((yn - ys)/d)

    return xc, yc

def init_local_net(cell, hop):
    global cells
    global cell_group

    vertex = []
    edge = []

    if len(cell_group[cell]['neighbors']) == 0:
        return vertex, edge

    if hop != 1:
        for neighbor in cell_group[cell]['neighbors']:
            sub_vertex, sub_edge = init_local_net(neighbor['cell'], hop - 1)
            for node in sub_vertex:
                if node not in vertex:
                    vertex.append(node)

            if cell_group[neighbor['cell']]['active'] not in vertex:
                vertex.append(cell_group[neighbor['cell']]['active'])
            
            for link in sub_edge:
                node1, node2 = link
                if (node1, node2) not in edge and (node2, node1) not in edge:
                    edge.append(link)
    else:
        vertex.append(cell_group[cell]['active'])
        for neighbor_1_hop in cell_group[cell]['neighbors']:
            vertex.append(neighbor_1_hop['node'])
            edge.append((neighbor_1_hop['node'], cell_group[cell]['active']))
        
    return vertex, edge

def find_e_res(cell, cell_node):
    global cells
    global cell_group

    e_res = None
    for node in cell_group[cell]['nodes']:
        if cell_node == node['pos']:
            e_res = node['energy']
            break

    if e_res is None:
        print("NO NODE FOUNDED IN CELL")
    
    return e_res

def e_cost_func(x):
    return np.exp(x/10)

def e_cost(tx_power, e_res):
    M = 0.1
    T = 1.0
    cost = integrate.quad(e_cost_func, e0 - e_res, e0 - e_res + tx_power * T)
    return cost[0] / M

def e_balance_benefit(cell_active):
    global cells
    global cell_group

    ALPHA = 1.5
    e_neighbors = []
    sum_e_neighbors = 0
    avg_e_neighbors = 0
    
    if len(cell_group[cell_active]['local_net']['nodes']) == 0:
        return 0

    # find residual energy of all neighbor nodes
    for net_node in cell_group[cell_active]['local_net']['nodes']:
        xc, yc = mapping_to_cell(net_node)              
        e_res = find_e_res((xc*50, yc*50), net_node)
        sum_e_neighbors += e_res
        e_neighbors.append(e_res)
    
    avg_e_neighbors = sum_e_neighbors * 1.0 / len(cell_group[cell_active]['local_net']['nodes'])
    
    # sum of difference between node in topology and average
    sum_diff = 0
    for e in e_neighbors:
        sum_diff += pow(e - avg_e_neighbors, 2)
    
    benefit = ALPHA * sum_diff / len(cell_group[cell_active]['local_net']['nodes'])
    return benefit

def ctb_benefit(cell_active):
    BETA = 1.5
    return BETA * len(cell_group[cell_active]['local_net']['nodes'])

# partioning of correlation region
for node in nodes:
    xn, yn = node
    xc, yc = mapping_to_cell(node)

    if (xc*50, yc*50) not in cells:
        cells.append((xc*50, yc*50))
        cell_group[(xc*50, yc*50)] = {
            'nodes': [],
            'active': None,
            'idle_time': 0,
            'neighbors': [],
            'local_net': {
                'nodes': [],
                'links': [],
            },
            'Vpre': None,
            'util': None,
            'power': p_max,
        }
    cell_node = {
        'pos': (xn, yn),
        'energy': e0,
    }
    cell_group[(xc*50, yc*50)]['nodes'].append(cell_node)

# start TC-GSC
actives_changed = False
while t < max_t:
    # initial phase
    actives = []
    for cell in cells:
        d_sqr = 0
        active_node = None
        if t == 0:
            for node in cell_group[cell]['nodes']:
                x, y = node['pos']
                if d_sqr == 0:
                    d_sqr = pow(x, 2) + pow(y, 2)
                    active_node = node
                elif (pow(x, 2) + pow(y, 2)) < d_sqr:
                    distance = (pow(x, 2) + pow(y, 2))
                    active_node = node
        elif cell_group[cell]['active'] is None:
            max_e_res = None
            for node in cell_group[cell]['nodes']:
                if max_e_res is None:
                    max_e_res = node['energy']
                    active_node = node
                elif node['energy'] > max_e_res:
                    max_e_res = node['energy']
                    active_node = node

        if (active_node != None):
            Vpre = random.uniform(2.8, 4.2)
            cell_group[cell]['active'] = active_node['pos']
            cell_group[cell]['Vpre'] = Vpre # not useful
            cell_group[cell]['idle_time'] = t + gamma * math.exp(Vsta/Vpre)
            cell_group[cell]['util'] = None
            cell_group[cell]['power'] = p_max
            actives.append(active_node['pos'])

    linking = []
    if actives_changed is True or t == 0:
        actives_changed = False
        for cell1 in cells:  # active node (xa, ya)
            cell_group[cell1]['neighbors'] = []
            for cell2 in cells: # all actives
                if cell1 == cell2:
                    continue
                active1_x, active1_y = cell_group[cell1]['active']
                active2_x, active2_y = cell_group[cell2]['active']
                r_x = active1_x - active2_x
                r_y = active1_y - active2_y
                p_transmit = cell_group[cell1]['power'] * pow(wave, 2) / (4 * pow(math.pi, 2) * (pow(r_x, 2) + pow(r_y, 2)))
                if (p_transmit >= pth):
                    neighbor = {
                        'node': cell_group[cell2]['active'],
                        'cell': cell2,
                    }
                    cell_group[cell1]['neighbors'].append(neighbor)
                    if ((active1_x, active1_y), (active2_x, active2_y)) not in linking and ((active2_x, active2_y), (active1_x, active1_y)) not in linking:
                        linking.append(((active1_x, active1_y), (active2_x, active2_y)))

        # create local topology of node i
        for cell in cells:
            cell_group[cell]['local_net']['nodes'], cell_group[cell]['local_net']['links'] = init_local_net(cell, hop_max)

    # plot.mapplot(nodes, area, actives, linking, None)

    # adaptation phase
    nash_eq = False
    while nash_eq is False:
        nash_eq = True
        for cell in cells:
            new_power = cell_group[cell]['power'] - p_step
            e_res_active = find_e_res(cell, cell_group[cell]['active'])
            if cell_group[cell]['util'] is None:
                cell_group[cell]['util'] = ctb_benefit(cell) - e_balance_benefit(cell) - e_cost(cell_group[cell]['power'], e_res_active)
            active_x, active_y = cell_group[cell]['active']
            new_util = None
            
            topology_changed = False
            old_neighbor = cell_group[cell]['neighbors']

            # check local topologies
            for neighbor in cell_group[cell]['neighbors']:
                neighbor_x, neighbor_y = neighbor['node']
                r_x = neighbor_x - active_x
                r_y = neighbor_y - active_y
                p_new_transmit = new_power * pow(wave, 2) / (4 * pow(math.pi, 2) * (pow(r_x, 2) + pow(r_y, 2)))
                if p_new_transmit < pth:
                    topology_changed = True
                    cell_group[cell]['neighbors'].remove(neighbor)      # change old neighbor

            new_local_net = {'nodes': [], 'links': []}
            new_local_net['nodes'], new_local_net['links'] = init_local_net(cell, hop_max)

            if len(new_local_net['nodes']) != len(cell_group[cell]['local_net']['nodes']):
                new_util = -1.0 * e_cost(p_new_transmit, e_res_active)
            else:
                # print('-------------------------------------------')
                # print('Cell: ', cell)
                # print('Contribute benefit: ', ctb_benefit(cell))
                # print('Energy balance benefit: ', e_balance_benefit(cell))
                # print('Cost: ', e_cost(cell_group[cell]['power'], e_res_active))
                new_util = ctb_benefit(cell) - e_balance_benefit(cell) - e_cost(new_power, e_res_active)
                    
            if new_util > cell_group[cell]['util']:
                cell_group[cell]['util'] = new_util
                cell_group[cell]['power'] = new_power
                nash_eq = False
                if topology_changed is True:
                    cell_group[cell]['local_net'] = new_local_net
            else:
                cell_group[cell]['neighbors'] = old_neighbor        # recover old neighbor if no power change

    actives = []
    for cell in cells:
        if cell_group[cell]['active'] is not None:
            actives.append(cell_group[cell]['active'])

    # maintainance phase
    new_actives = []
    for cell in cells:
        e_res_active = None
        for node in cell_group[cell]['nodes']:
            if node['pos'] == cell_group[cell]['active']:
                node['energy'] -= cell_group[cell]['power']         # consume power in each iteration
                e_res_active = node['energy']

        if cell_group[cell]['idle_time'] < t or e_res_active < eth:
            cell_group[cell]['active'] = None
            actives_changed = True

    # log active nodes' power
    power = [round(cell_group[cell]['power'], 4) for cell in cells]
    print('-------------------------------------------')
    print(f'Iteration {t}:')
    print(power)
    
    t += 1

linking = []
for cell in cells:
    for link in cell_group[cell]['local_net']['links']:
        node1, node2 = link
        if (node1, node2) not in linking and (node2, node1) not in linking:
            linking.append(link)

e_res_list = []
for cell in cells:
    for node in cell_group[cell]['nodes']:
        e_res_list.append({
            'node': node['pos'],
            'energy': node['energy'],
        })
        
print('-------------------------------------------')
print('Residual energy node:')
for item in e_res_list:
    print(item)
plot.mapplot(nodes, area, actives, linking, None)