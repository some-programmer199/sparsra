import jax
import jax.numpy as jnp
from jax import random

# =====================================================
# CPPN Genome Network
# =====================================================

def cppn_single(params, neuron_idx):
    """Generate one neuron's properties from genome (CPPN)."""
    x = jnp.array([neuron_idx / 50])  # scaled idx (remember to change)
    h1 = jnp.tanh(x @ params["W1"] + params["b1"])
    
    h2 = jnp.sin(h1 @ params["W2"] + params["b2"])
    h3 = jax.nn.sigmoid(h2 @ params["W3"] + params["b3"])
    
    coords = jnp.tanh(h3[:2])                      
    axon = jnp.tanh(h3[2:4] + coords * 0.5)        
    weights = jnp.tanh(h3[4:8])                    
    spike_vec = jnp.tanh(h3[8:12])                 
    radius = jax.nn.softplus(h3[12])  
    threshold = jax.nn.softplus(h3[13]) + 0.1       # avoid zero threshold             
    
    return coords, axon, weights, spike_vec, radius,threshold

cppn_batch = jax.vmap(cppn_single, in_axes=(None, 0), out_axes=0)

def init_cppn_params(key, hidden=128):
    k1, k2, k3 = random.split(key, 3)
    return {
        "W1": random.normal(k1, (1, hidden)) * 0.2,
        "b1": jnp.zeros(hidden),
        "W2": random.normal(k2, (hidden, hidden)) * 0.2,
        "b2": jnp.zeros(hidden),
        "W3": random.normal(k3, (hidden, 20)) * 0.2,
        "b3": jnp.zeros(20),
    }

# =====================================================
# Agent Development
# =====================================================

def develop_agent(params, num_neurons):
    """Develop agent: generate neurons and build connections."""
    idxs = jnp.arange(num_neurons)
    coords, axons, weights, spike_vecs, radii, threshold= cppn_batch(params, idxs)
    
    axon_tips = coords + axons * radii[:, None]
    dists = jnp.linalg.norm(axon_tips[:, None, :] - coords[None, :, :], axis=-1)
    
    mask = dists < radii[:, None]
    src, dst = jnp.where(mask)
    mask_valid = src != dst
    src, dst = src[mask_valid], dst[mask_valid]
    
    return {
        "weights": weights,              # (N,4)
        "spike_vecs": spike_vecs,        # (N,4)
        "threshold": threshold,
        "decay": jnp.ones(num_neurons) * 0.95,
        "reset": jnp.zeros(num_neurons), # reset voltage after spike
        "connections": {"src": src, "dst": dst}
    }

# =====================================================
# LIF SNN Step with Refractory Period
# =====================================================

def snn_step(state, spikes, refractory, agent,inputs, refractory_steps=2,dt=0):
    state += inputs
    
    agent["threshold"]=agent["threshold"]+jnp.sin(dt/jnp.pi*2)*0.2
    """
    Run 1 timestep of LIF spiking network with refractory.
    
    state: (N,) membrane potentials
    spikes: (N,) bool spikes from previous step
    refractory: (N,) int steps remaining in refractory
    """
    N = state.shape[0]
    src = agent["connections"]["src"]
    dst = agent["connections"]["dst"]
    
    # Emit spike vectors from neurons not in refractory
    active = (refractory == 0)
    emitted = agent["spike_vecs"] * spikes[:, None] * active[:, None]  # (N,4)
    
    incoming = emitted[src]                        # (E,4)
    receiver_w = agent["weights"][dst]             # (E,4)
    contrib = jnp.sum(incoming * receiver_w, axis=-1)  # (E,)
    
    # Aggregate contributions
    delta = jnp.zeros(N).at[dst].add(contrib)
    
    # Update voltages with decay
    v = state * agent["decay"] + delta
    
    # Determine new spikes (only if not refractory)
    new_spikes = (v > agent["threshold"]) & (refractory == 0)
    
    # Reset voltage of neurons that spiked
    v = jnp.where(new_spikes, agent["reset"], v)
    
    # Update refractory counters
    refractory = jnp.where(new_spikes, refractory_steps, jnp.maximum(0, refractory - 1))
    
    return v, new_spikes, refractory

# =====================================================
# Example
# =====================================================

if __name__ == "__main__":
    key = random.PRNGKey(42)
    params = init_cppn_params(key)
    agent = develop_agent(params, num_neurons=50)
    inputs=jnp.zeros(50)
    inputs=inputs.at[:50].set(10.0)
    state = jnp.zeros(50)
    spikes = jnp.zeros(50, dtype=bool)
    refractory = jnp.zeros(50, dtype=int)
    
    for t in range(10):
        state, spikes, refractory = snn_step(state, spikes, refractory, agent,inputs=inputs, refractory_steps=2)
        inputs=jnp.zeros(50)
        inputs=inputs.at[:10].set(10.0)
        print(f"Timestep {t}: spikes={spikes.sum()}, avgV={state.mean():.3f}, refractory={refractory.sum()}")