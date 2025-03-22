import matplotlib.pyplot as plt

def mapplot(nodes, area, actives, linking, local_net):
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]

    x_actives = [active[0] for active in actives]
    y_actives = [active[1] for active in actives]

    plt.figure(figsize=(10, 10))  # Set figure size
    plt.scatter(x_coords, y_coords, c='blue', marker='o', s=50)  # Plot nodes as blue dots
    plt.scatter(x_actives, y_actives, c='red', marker='o', s=40)  # Plot nodes as blue dots

    for link in linking:    
        node1, node2 = link
        x11, x12 = node1
        x21, x22 = node2
        plt.plot([x11, x21], 
                [x12, x22], 
                color='black',
                linewidth=1)
        
    # draw local network
    if local_net is not None:
        x_local = [node[0] for node in local_net['nodes']]
        y_local = [node[1] for node in local_net['nodes']]
        for link in local_net['links']:    
            node1, node2 = link
            x11, x12 = node1
            x21, x22 = node2
            plt.plot([x11, x21], 
                    [x12, x22], 
                    color='orange',
                    linewidth=0.5)

        plt.scatter(x_local, y_local, c='green', marker='o', s=20)

    # Set axis limits
    plt.xlim(-area * 1.2, area * 1.2)
    plt.ylim(-area * 1.2, area * 1.2)

    # Add grid
    plt.grid(True, 
         which='both',  # Show both major and minor grid lines
         linestyle='--',  # Dashed lines
         color='gray',    # Grid color
         alpha=0.5,       # Transparency (0-1)
         linewidth=0.5)   # Line thickness

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Map')

    # Set custom grid spacing
    plt.xticks(range(-area, area + 1, int(area/5)))  # Major grid lines every 20 units on x-axis
    plt.yticks(range(-area, area + 1, int(area/5)))  # Major grid lines every 20 units on y-axis

    # Add origin lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Show the plot
    plt.show()