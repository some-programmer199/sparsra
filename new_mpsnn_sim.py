import jax
import jax.numpy as jnp
from jax import random


# ----------------------
# GENOME NETWORK (CPPN style)
# ----------------------

def cppn(params, neuron_idx):
    """CPPN-like genome network to generate neuron properties."""
    # Input: neuron index encoded as float
    x = jnp.array([neuron_idx / 10000.0])  # scale idx to [0,1]
    
    # Layer 1
    h1 = jnp.tanh(x @ params["W1"] + params["b1"])
    # Layer 2 with sine nonlinearity
    h2 = jnp.sin(h1 @ params["W2"] + params["b2"])
    # Layer 3 with sigmoid
    h3 = jax.nn.sigmoid(h2 @ params["W3"] + params["b3"])

    # Outputs
    coords = jnp.tanh(h3[:2])                      # neuron position
    axon = jnp.tanh(h3[2:4] + coords * 0.5)        # axon direction
    weights = jnp.tanh(h3[4:8])                    # receiver weights (4 vals)
    spike_vec = jnp.tanh(h3[8:12])                 # neuron’s spike vector
    radius = jax.nn.softplus(h3[12])               # connection radius

    return coords, axon, weights, spike_vec, radius


def init_cppn_params(key, hidden=300):
    """Initialize weights for a CPPN-like genome net."""
    k1, k2, k3 = random.split(key, 3)
    return {
        "W1": random.normal(k1, (1, hidden)) * 0.2,
        "b1": jnp.zeros(hidden),
        "W2": random.normal(k2, (hidden, hidden)) * 0.2,
        "b2": jnp.zeros(hidden),
        "W3": random.normal(k3, (hidden, 20)) * 0.2,  # enough outputs
        "b3": jnp.zeros(20),
    }


# ----------------------
# SNN STEP (your dynamics)
# ----------------------

def snn_step(state, spikes, neuron_weights, spike_vecs, stats, connections):
    """
    Run 1 timestep of your custom SNN.

    state: (N,) voltages
    spikes: (N,) last timestep spikes (bool/int)
    neuron_weights: (N,4) each neuron's receiver weights
    spike_vecs: (N,4) each neuron's own spike vector
    stats: dict with 'threshold' (N,), 'decay' (N,)
    connections: dict with 'src' (E,), 'dst' (E,)
    """
    N = state.shape[0]
    E = connections["src"].shape[0]

    # Which neurons spiked last step → emit their spike vectors
    emitted = spike_vecs * spikes[:, None]  # (N,4)

    # Gather per-connection incoming spikes
    incoming = emitted[connections["src"]]  # (E,4)
    dst = connections["dst"]

    # Each receiving neuron applies its own weights
    receiver_w = neuron_weights[dst]        # (E,4)
    contrib = jnp.sum(incoming * receiver_w, axis=-1)  # (E,)

    # Accumulate contributions into target voltages
    delta = jnp.zeros(N).at[dst].add(contrib)

    # Update voltages with decay
    new_state = state * stats["decay"] + delta

    # New spikes
    new_spikes = new_state > stats["threshold"]

    return new_state, new_spikes
