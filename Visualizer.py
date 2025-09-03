import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Agents import make_example_genome, develop_agent

def visualize_agent(agent, show_radius=True, max_neurons=100, figsize=(10, 10)):
    """
    Visualize an agent's neural layout and connections.
    
    Args:
        agent: dict with 'neurons' array [soma_x, soma_y, ax_x, ax_y, radius]
        show_radius: bool, whether to show connection radii
        max_neurons: int, maximum number of neurons to visualize (samples if more)
        figsize: tuple, figure dimensions
    """
    neurons = agent['neurons']
    N = len(neurons)
    
    # if too many neurons, sample randomly
    if N > max_neurons:
        idx = np.random.choice(N, max_neurons, replace=False)
        neurons = neurons[idx]
        N = max_neurons
    
    # extract positions and radii
    soma_pos = neurons[:, :2]
    axon_pos = neurons[:, 2:4]
    radii = neurons[:, 4]
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot somas as dots
    ax.scatter(soma_pos[:, 0], soma_pos[:, 1], c='blue', s=50, alpha=0.6, label='Somas')
    
    # plot axon endpoints as x's
    ax.scatter(axon_pos[:, 0], axon_pos[:, 1], c='red', marker='x', s=50, alpha=0.6, label='Axons')
    
    # draw lines from soma to axon for each neuron
    for i in range(N):
        ax.plot([soma_pos[i, 0], axon_pos[i, 0]], 
                [soma_pos[i, 1], axon_pos[i, 1]], 
                'gray', alpha=0.2, linewidth=0.5)
    
    # show connection radii if requested
    if show_radius:
        for i in range(N):
            circle = plt.Circle((axon_pos[i, 0], axon_pos[i, 1]), 
                              radii[i], color='red', fill=False, 
                              alpha=0.1, linestyle='--')
            ax.add_patch(circle)
    
    # compute connections (similar to develop_agent logic)
    diffs = soma_pos[None, :, :] - axon_pos[:, None, :]  # (N, N, 2)
    distances = np.sqrt(np.sum(diffs**2, axis=-1))  # (N, N)
    triu = np.triu(np.ones_like(distances), k=1)  # upper triangular mask (i<j)
    connections = (distances <= radii[:, None]) & (triu > 0)
    
    # draw connections
    from_idx, to_idx = np.where(connections)
    for i, j in zip(from_idx, to_idx):
        ax.plot([axon_pos[i, 0], soma_pos[j, 0]], 
                [axon_pos[i, 1], soma_pos[j, 1]], 
                'green', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Neural Layout (showing {N} of {len(agent["neurons"])} neurons)')
    ax.legend()
    plt.axis('equal')
    return fig, ax

if __name__ == "__main__":
    # Create and visualize an example agent
    genome = make_example_genome(n_neurons=1000, n_connections=5000, seed=42)
    agent = develop_agent("test", genome, n_neurons=1000, normalize=True)
    
    # Create two visualizations: with and without radii
    fig1, ax1 = visualize_agent(agent, show_radius=True, max_neurons=50)
    plt.savefig('neural_layout_with_radii.png', dpi=300, bbox_inches='tight')
    
    fig2, ax2 = visualize_agent(agent, show_radius=False, max_neurons=50)
    plt.savefig('neural_layout_no_radii.png', dpi=300, bbox_inches='tight')
    
    plt.show()