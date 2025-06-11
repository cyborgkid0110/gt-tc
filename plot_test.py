import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

def build_graph(vertices, edges):
    """Build a directed graph from vertices and adjacency matrix."""
    G = nx.DiGraph()
    index_map = {vertex: idx for idx, vertex in enumerate(vertices)}
    
    for i, u in enumerate(vertices):
        G.add_node(u)
        for j, v in enumerate(vertices):
            if edges[i][j] == 1:
                G.add_edge(u, v)
    return G

def get_clusters(node_dict):
    """Group nodes by their CH_belong value."""
    clusters = defaultdict(list)
    for node in node_dict:
        ch = node_dict[node]['CH_belong']
        if ch is not None:
            clusters[ch].append(node)
        else:
            clusters["unclustered"].append(node)
    return clusters

def assign_layers_to_cluster(G, cluster_nodes, ch_node):
    """
    Assign each node in the cluster to a layer based on minimum hop distance from CH.
    Returns a list of lists: layers[layer_index] = [nodes]
    """
    layers = defaultdict(list)
    visited = set()
    queue = deque()

    layers[0] = [ch_node]
    visited.add(ch_node)
    queue.append((ch_node, 0))

    while queue:
        current, dist = queue.popleft()
        for neighbor in G.successors(current):
            if neighbor in cluster_nodes and neighbor not in visited:
                visited.add(neighbor)
                layers[dist + 1].append(neighbor)
                queue.append((neighbor, dist + 1))

    # Sort layers by key to ensure order
    sorted_layers = [layers[key] for key in sorted(layers.keys())]
    return sorted_layers

def split_into_batches(layer_nodes, G):
    """
    Split nodes in a layer into batches where nodes are weakly connected.
    """
    subG = G.subgraph(layer_nodes).copy()
    batches = []
    visited = set()

    for node in layer_nodes:
        if node not in visited:
            component = nx.single_source_shortest_path(subG.to_undirected(), node).keys()
            batch = [n for n in layer_nodes if n in component]
            batches.append(batch)
            visited.update(batch)
    return batches

def divide_network_by_clusters(G, node_dict):
    """
    Main function to divide network into layered batches per cluster.
    Returns a dict: {ch_position: [[batch1, batch2,...], [layer2_batches,...], ...]}
    """
    clusters = get_clusters(node_dict)
    layered_batches_per_cluster = {}

    for ch_pos, members in clusters.items():
        if ch_pos == "unclustered":
            continue
        if not node_dict[ch_pos]['CH']:
            continue  # Skip invalid CHs

        cluster_subgraph_nodes = [node for node in members if node in G.nodes]
        layered_structure = assign_layers_to_cluster(G, cluster_subgraph_nodes, ch_pos)

        layered_batches = []
        for layer_nodes in layered_structure:
            batches = split_into_batches(layer_nodes, G)
            layered_batches.append(batches)

        layered_batches_per_cluster[ch_pos] = layered_batches

    return layered_batches_per_cluster

def draw_network(G, layered_batches_per_cluster, node_dict):
    pos = {node: node for node in G.nodes}
    plt.figure(figsize=(10, 8))

    # Draw all edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Prepare color map
    color_map = []
    for node in G.nodes:
        if node_dict[node]['CH']:
            color_map.append('red')
        else:
            color_map.append('blue')

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=100)

    # Optionally label CHs
    labels = {node: 'CH' for node in G.nodes if node_dict[node]['CH']}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

    plt.title("Wireless Sensor Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example Usage:
if __name__ == "__main__":
    # Sample data
    network = {
        'vertices': [(0, 0), (1, 1), (2, 2), (1, 0), (0, 1), (2, 1)],
        'edges': [
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    }

    node_dict = {
        (0, 0): {'CH': True, 'CH_belong': None, 'rc': 0.05, 'neighbor': [(1, 1), (1, 0)]},
        (1, 1): {'CH': False, 'CH_belong': (0, 0), 'rc': 0.07, 'neighbor': [(2, 2), (1, 0), (2, 1)]},
        (2, 2): {'CH': False, 'CH_belong': (0, 0), 'rc': 0.06, 'neighbor': [(2, 1)]},
        (1, 0): {'CH': False, 'CH_belong': (0, 0), 'rc': 0.04, 'neighbor': [(0, 0), (0, 1)]},
        (0, 1): {'CH': False, 'CH_belong': (0, 0), 'rc': 0.05, 'neighbor': []},
        (2, 1): {'CH': False, 'CH_belong': (0, 0), 'rc': 0.05, 'neighbor': []},
    }

    G = build_graph(network['vertices'], network['edges'])
    layered_batches_per_cluster = divide_network_by_clusters(G, node_dict)
    
    # Print layered batches result
    for ch_pos, layers in layered_batches_per_cluster.items():
        print(f"\nCluster headed by {ch_pos}:")
        for i, layer in enumerate(layers):
            print(f"Layer {i}: {layer}")

    # Draw the graph
    draw_network(G, layered_batches_per_cluster, node_dict)