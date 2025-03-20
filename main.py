import random
import matplotlib.pyplot as plt
import plot
import math

# # Set random seed for reproducibility (optional)
# random.seed()

# Create list to store nodes
nodes = []
num_nodes = 200
area = 250
xs, ys = (0, 0) # sink node

e0 = 50      # initial energy of the nodes
pth = 7*pow(10, -9)     # power threshold
hop_max = 3     # number of hops
d = 50      # cell size
Vsta = 3.6  # standard working voltage
gamma = 5
p_strat = []
p_min = 0.01
p_max = 0.1
wave = 0.1224

# Generate 50 nodes
for i in range(num_nodes):
    x = random.uniform(-area, area)  # Random x coordinate between -100 and 100
    y = random.uniform(-area, area)  # Random y coordinate between -100 and 100
    nodes.append((round(x, 1), round(y, 1)))

cells = []
# cell_pos = [
#     (0, 0),...
# ]
cell_group = {}
# cells = {
#     (0, 0): {
#         'nodes': [{
#               'pos': (0, 0),
#               'energy': 0,
#         },...],
#         'active': None,
#         'energy': 0,
#         'idle_time': 0,
#         'power': 0, 
#         'neighbors': {
#             1: [{
#                 'cell': (0,0),
#                 'node': (0,0),
#             },...],
#             2: ...
#         },
#     },
#     ...
# }

# partioning of correlation region
for node in nodes:
    xn, yn = node
    if xn > xs:
        xc = math.ceil((xn - xs)/d)
    else:
        xc = math.floor((xn - xs)/d)

    if yn > ys:
        yc = math.ceil((yn - ys)/d)
    else:
        yc = math.floor((yn - ys)/d)

    if (xc*50, yc*50) not in cells:
        cells.append((xc*50, yc*50))
        cell_group[(xc*50, yc*50)] = {
            'nodes': [],
            'active': None,
            'idle_time': 0,
            'neighbors': {
                1: []
            },
            'strategy': [],
            'power': p_max,
        }
    cell_node = {
        'pos': (xn, yn),
        'energy': e0,
    }
    cell_group[(xc*50, yc*50)]['nodes'].append(cell_node)
    
# initial phase
for cell in cells:
    d_sqr = 0
    active_node = None
    for node in cell_group[cell]['nodes']:
        x, y = node['pos']
        if d_sqr == 0:
            d_sqr = pow(x, 2) + pow(y, 2)
            active_node = node
        elif (pow(x, 2) + pow(y, 2)) < d_sqr:
            distance = (pow(x, 2) + pow(y, 2))
            active_node = node

    if (active_node != None):
        Vpre = random.uniform(2.8, 4.2)
        cell_group[cell]['active'] = active_node['pos']
        cell_group[cell]['idle_time'] = gamma * math.exp(Vsta/Vpre)

actives = []
for cell in cells:
    if cell_group[cell]['active'] != None:
        actives.append(cell_group[cell]['active'])

t = 0
linking = []
for cell1 in cells:  # active node (xa, ya)
    for cell2 in cells: # all actives
        if cell1 == cell2:
            continue
        active1_x, active1_y = cell_group[cell1]['active']
        active2_x, active2_y = cell_group[cell2]['active']
        r_x = active1_x - active2_x
        r_y = active1_y - active2_y
        p_transmit = cell_group[cell1]['power'] * pow(wave, 2) / (4 * pow(math.pi, 2) * (pow(r_x, 2) + pow(r_y, 2)))
        if (p_transmit > pth):
            neighbor = {
                'node': cell_group[cell2]['active'],
                'cell': cell2,
            }
            cell_group[cell1]['neighbors'][1].append(neighbor)
            # print("Debug:", active1_x, active1_y, active2_x, active2_y)
            # print(f"{(pow(r_x, 2) + pow(r_y, 2))}, {p_transmit}")
            if ((active1_x, active1_y), (active2_x, active2_y)) not in linking and ((active2_x, active2_y), (active1_x, active1_y)) not in linking:
                linking.append(((active1_x, active1_y), (active2_x, active2_y)))

# 2-hop
for hop in range(2, hop_max+1):
    for cell1 in cells:
        cell_group[cell1]['neighbors'][hop] = []
        for neighbor in cell_group[cell]['neighbors'][hop-1]:
            cell2 = neighbor['cell']
            for remote_neighbor in cell_group[cell2]['neighbors'][1]:
                cell_group[cell1]['neighbors'][hop].append(remote_neighbor)

# for cell in cells:
#     print(f'Neighbor of cell {cell_group[cell]['active']}: {cell_group[cell]['neighbors']}')
#     print('\n')

# print(linking)
plot.mapplot(nodes, area, actives, linking)