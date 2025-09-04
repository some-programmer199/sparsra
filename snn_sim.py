import jax
import jax.numpy as jnp
from jax.experimental.sparse import bcoo
import Agents

# Message-passing SNN: each neuron has a membrane potential, receives weighted spikes, and fires a vector if above threshold

def build_bcoo_matrix(genome, n_neurons):
    """
    Build a JAX BCOO sparse matrix from edge list genome.
    genome: array-like (E,3) [from, to, weight]
    Returns: bcoo.BCOO of shape (n_neurons, n_neurons)
    """
    edges = jnp.asarray(genome)
    if edges.ndim != 2 or edges.shape[1] not in (2, 3):
        raise ValueError("genome edges must be shape (E,2) or (E,3)")
    idx = edges[:, :2].astype(jnp.int32)
    if edges.shape[1] == 3:
        w = edges[:, 2].astype(jnp.float32)
    else:
        w = jnp.ones((idx.shape[0],), dtype=jnp.float32)
    mat = bcoo.BCOO((w, idx), shape=(n_neurons, n_neurons))
    return mat


def sim_mpsnn(
    genome,
    n_neurons=1000,
    n_steps=10,
    input_fn=None,
    threshold=1.0,
    reset=0.0,
    initial_state=None
):
    """
    Message-passing SNN simulation using JAX and BCOO.
    Each neuron has a membrane potential. Incoming spikes are weighted and summed.
    Neuron fires if membrane potential exceeds threshold.
    Args:
        genome: edge list [from, to, weight]
        n_neurons: number of neurons
        n_steps: number of simulation steps
        input_fn: function(step, state) -> input vector (n_neurons,) or None
        threshold: firing threshold
        reset: value to reset membrane after firing
    Returns:
        states: (n_steps+1, n_neurons) membrane potentials over time
        spikes: (n_steps, n_neurons) spike events (bool)
    """
    conn = build_bcoo_matrix(genome, n_neurons)
    # allow caller to provide a starting state
    if initial_state is None:
        state0 = jnp.zeros(n_neurons, dtype=jnp.float32)
    else:
        state0 = jnp.asarray(initial_state, dtype=jnp.float32)

    def step_fn(carry, t):
        state = carry
        # External input
        input_vec = input_fn(t, state) if input_fn is not None else jnp.zeros_like(state)
        # Ensure input_vec is an array and compatible with state shape.
        input_vec = jnp.asarray(input_vec, dtype=jnp.float32)
        if input_vec.shape != state.shape:
            # allow short vectors to be injected into the first indices (e.g., length 2 -> neurons [0,1])
            if input_vec.ndim == 0:
                input_vec = jnp.full_like(state, input_vec)
            elif input_vec.ndim == 1 and input_vec.shape[0] <= state.shape[0]:
                buf = jnp.zeros_like(state)
                buf = buf.at[: input_vec.shape[0]].set(input_vec)
                input_vec = buf
            else:
                raise ValueError(f"input_fn returned shape {input_vec.shape} incompatible with state shape {state.shape}")
        # Compute incoming spikes (who fired last step)
        presyn_spikes = state > threshold
        # Message passing: weighted sum of presynaptic spikes
        incoming = conn @ presyn_spikes.astype(jnp.float32)
        # Update membrane potential
        new_state = state + incoming + input_vec
        # Neuron fires if above threshold
        spikes = new_state > threshold
        # Reset fired neurons
        new_state = jnp.where(spikes, reset, new_state)
        return new_state, spikes

    # Run scan for n_steps
    states = [state0]
    spikes_list = []
    state = state0
    for t in range(n_steps):
        state, spikes = step_fn(state, t)
        states.append(state)
        spikes_list.append(spikes)
    states = jnp.stack(states)
    spikes = jnp.stack(spikes_list)
    return states, spikes

# Example usage:
if __name__ == "__main__":
    # Create a random genome
    genome = Agents.make_example_genome(n_neurons=100, n_connections=300, seed=42)
    print("Genome shape:", genome.shape)
    # Simple input: inject 1.5 to neuron 0 at t=0
    def input_fn(t, state):
        return jnp.where((t==0), jnp.eye(1, 100, 0).flatten()*1.5, jnp.zeros(100))
    states, spikes = sim_mpsnn(genome, n_neurons=100, n_steps=20, input_fn=input_fn)
    print("states shape:", states.shape)
    print("spikes shape:", spikes.shape)
    print("Final membrane potentials:", states[-1])
