import jax
import jax.numpy as jnp
import snn_sim

agents={}


def normalize_genome(genome, n_neurons: int = 1000, n_connections: int = 5000, seed: int = 0):
    """
    Ensure genome is feedforward and standardized to (n_connections, 3).

    - genome: array-like with rows [from, to, weight]
    - Enforces to_idx >= from_idx (feedforward) by filtering out edges that violate it.
    - Clips indices to [0, n_neurons-1].
    - If there are fewer than n_connections after filtering, pads with randomly
      generated feedforward edges (uniform weights in [-1,1]) using `seed`.
    - If there are more, truncates to n_connections.
    Returns a jnp.array of shape (n_connections, 3) dtype float32.
    """
    edges = jnp.asarray(genome, dtype=jnp.float32)
    if edges.ndim == 1:
        # allow empty or single-row flattened
        if edges.size == 0:
            edges = edges.reshape((0, 3))
        else:
            raise ValueError("genome must be shape (E,3)")
    if edges.ndim != 2 or edges.shape[1] < 3:
        raise ValueError("genome must be shape (E,3) with rows [from,to,weight]")

    # extract and sanitize
    from_idx = edges[:, 0].astype(jnp.int32)
    to_idx = edges[:, 1].astype(jnp.int32)
    weights = edges[:, 2].astype(jnp.float32)

    # clip to valid range
    from_idx = jnp.clip(from_idx, 0, n_neurons - 1)
    to_idx = jnp.clip(to_idx, 0, n_neurons - 1)

    # keep only feedforward edges (to >= from)
    mask = to_idx >= from_idx
    ff_from = from_idx[mask]
    ff_to = to_idx[mask]
    ff_w = weights[mask]

    edges_ff = jnp.stack([ff_from.astype(jnp.float32), ff_to.astype(jnp.float32), ff_w], axis=1)

    cur = edges_ff.shape[0]
    if cur >= n_connections:
        return edges_ff[:n_connections, :].astype(jnp.float32)

    # need padding: create random feedforward edges
    need = int(n_connections - cur)
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)
    rand_from = jax.random.randint(k1, (need,), 0, n_neurons, dtype=jnp.int32)
    # sample offsets per-element so that to >= from
    # compute maximum allowed offset per element
    max_off = (n_neurons - 1 - rand_from).astype(jnp.float32)
    # uniform [0,1) then scale
    offs = jax.random.uniform(k2, (need,), minval=0.0, maxval=1.0)
    rand_to = rand_from + (offs * max_off).astype(jnp.int32)
    rand_w = jax.random.uniform(k3, (need,), minval=-1.0, maxval=1.0)

    pad = jnp.stack([rand_from.astype(jnp.float32), rand_to.astype(jnp.float32), rand_w.astype(jnp.float32)], axis=1)
    out = jnp.concatenate([edges_ff, pad], axis=0)
    return out.astype(jnp.float32)

def develop_agent(name, genome, n_neurons: int = 10000, normalize: bool = True,
                  net_n_neurons: int = 1000, net_steps: int = 10):
    """
    Build an agent with `n_neurons` by running a network-encoded `genome`
    (edge list) through `run_genome` for each neuron index.

    - genome: array-like edges (E,3) rows [from, to, weight] used by run_genome
    - For each neuron i in [0..n_neurons-1] we form a 2-element input:
        data = [idx_norm, idx_norm**2] where idx_norm = i / (n_neurons-1)
      and call run_genome(genome, data, n_neurons=net_n_neurons, n_steps=net_steps).
      The run_genome return (4,) is taken as [soma_x, soma_y, ax_x, ax_y].
    - Outputs are squashed to [0,1] with sigmoid by default.
    Returns an agent dict with 'neurons' as a jnp.ndarray shape (n_neurons, 4).
    """
    N = int(n_neurons)
    if N <= 0:
        raise ValueError("n_neurons must be positive")

    # normalize/standardize genome to feedforward 1000-neuron network with 5000 connections
    genome_arr = normalize_genome(genome, n_neurons=int(net_n_neurons), n_connections=5000, seed=0)

    def _mapper(idx):
        # Create highly nonlinear input features with discontinuities
        idx_norm = idx / jnp.maximum(N - 1, 1)
        
        # Generate a pseudo-random phase offset using a high-frequency sine
        phase = jnp.sin(idx_norm * 1000.0) * 10.0
        
        features = jnp.array([
            idx_norm,  # linear component
            jnp.sin(idx_norm * 31.41592 + phase),  # high freq with phase shift
            jnp.sign(jnp.sin(idx_norm * 7.0)) * jnp.abs(jnp.tan(idx_norm * 3.14159)),  # discontinuous
            jnp.floor(idx_norm * 5.0) / 5.0,  # step function
            jnp.where(jnp.sin(idx_norm * 20.0) > 0, 1.0, -1.0),  # binary switching
        ], dtype=jnp.float32)
        
        # Create two very different input signals
        data = jnp.array([
            features[0] + 0.3 * features[1] + 0.2 * features[2],  # smooth + chaos
            features[3] + 0.5 * features[4]  # discrete jumps
        ], dtype=jnp.float32)
        
        # Let the network run longer with these complex inputs
        out = run_genome(genome_arr, data, n_neurons=int(net_n_neurons), n_steps=int(net_steps) * 2)
        return out  # no scaling needed, network can now produce diverse outputs

    # vectorize across neuron indices
    mapper_v = jax.vmap(_mapper, in_axes=0)

    indices = jnp.arange(N, dtype=jnp.float32)
    raw_outputs = mapper_v(indices)  # expected (N, 5)

    # ensure shape (N, 5) for [soma_x, soma_y, ax_x, ax_y, radius]
    raw = jnp.reshape(raw_outputs, (N, -1))
    if raw.shape[1] != 5:
        raise ValueError(f"Genome/run_genome must produce 5 values per neuron (got {raw.shape[1]})")

    # Normalize soma positions, axon offsets, and radii separately
    if normalize:
        neurons = []
        
        # 1. Handle soma positions (first 2 columns) - full range
        for i in range(2):
            col = raw[:, i]
            low, high = jnp.percentile(col, jnp.array([5, 95]))
            scaled = (col - low) / (high - low + 1e-6)
            neurons.append(scaled)
            
        # 2. Handle axon positions relative to soma
        soma_pos = jnp.stack([neurons[0], neurons[1]], axis=1)
        axon_raw = raw[:, 2:4]
        
        # Calculate axon offsets from soma
        offsets = axon_raw - raw[:, :2]
        
        # Scale offsets nonlinearly to encourage shorter connections
        # but allow some longer ones with decreasing probability
        offset_lens = jnp.sqrt(jnp.sum(offsets**2, axis=1))
        low, high = jnp.percentile(offset_lens, jnp.array([5, 95]))
        lens_norm = (offset_lens - low) / (high - low + 1e-6)
        
        # Apply nonlinear scaling that favors shorter connections
        # Using a combination of linear and exponential decay
        scale_factor = 0.2 * jnp.exp(-2.0 * lens_norm) + 0.1 * (1.0 - lens_norm)
        scale_factor = jnp.clip(scale_factor, 0.02, 0.3)  # Allow some longer connections
        
        # Apply scaling to normalized offsets
        scaled_offsets = offsets / (offset_lens[:, None] + 1e-6) * scale_factor[:, None]
        
        # Convert back to absolute positions
        axon_pos = soma_pos + scaled_offsets
        neurons.extend([axon_pos[:, 0], axon_pos[:, 1]])
            
        # 3. Handle radius (last column) - make it small and always positive
        radii = raw[:, 4]
        low, high = jnp.percentile(radii, jnp.array([5, 95]))
        radii_norm = (radii - low) / (high - low + 1e-6)
        radii_scaled = jax.nn.sigmoid(radii_norm) * 0.15
        neurons.append(radii_scaled)
        
        neurons = jnp.stack(neurons, axis=1)
    else:
        neurons = raw

    # Generate connections based on spatial proximity
    positions_from = neurons[:, :2]  # soma positions
    axons_to = neurons[:, 2:4]  # axon positions
    radii = neurons[:, 4]  # connection radii

    # compute pairwise distances between axons and somas
    diffs = jax.vmap(lambda ax: positions_from - ax)(axons_to)
    distances = jnp.sqrt(jnp.sum(diffs**2, axis=-1))  # (N, N)
    
    # generate connections where distance <= radius
    # note: this maintains feedforward property since we only check from i->j where i<j
    triu = jnp.triu(jnp.ones_like(distances), k=1)  # upper triangular mask (i<j)
    radii_broadcast = radii[:, None]  # (N, 1)
    connections_mask = (distances <= radii_broadcast) & (triu > 0)
    
    # get connection indices and generate random weights
    from_idx, to_idx = jnp.where(connections_mask)
    n_valid = len(from_idx)
    key = jax.random.PRNGKey(0)
    weights = jax.random.uniform(key, (n_valid,), minval=-1.0, maxval=1.0)
    
    # pack into edges array and normalize to desired connection count
    edges = jnp.stack([from_idx, to_idx, weights], axis=1)

    agent = {
        "name": name,
        "genome": genome_arr,
        "age": 0,
        "score": 0,
        "neurons": neurons,  # shape (N,5): [soma_x, soma_y, ax_x, ax_y, radius]
        "spatial_edges": edges.astype(jnp.float32)  # (E,3) [from,to,weight]
    }
    agents[name] = agent
    return agent
  # should be (10000, 4)
#genome is a list or array with [from,to,weight] lists, assume there are 1000 neurons in the genome network, 2 neurons are where the input is given, 4 neurons are monitered for output
def run_genome(genome, data, n_neurons: int = 1000, n_steps: int = 10,
               input_idxs=(0, 1), output_idxs=None):
    """
    Run a genome given as edges [from, to, weight].

    - genome: array-like (E,3) rows [from_idx, to_idx, weight]
    - data: shape (2,) values to inject at input_idxs each step
    - n_neurons: total number of neurons in the simulated network (default 1000)
    - n_steps: number of synchronous propagation steps to run
    - input_idxs: tuple/list of indices receiving `data` (default (0,1))
    - output_idxs: indices to return; if None, returns the last 4 neurons

    Returns: jnp.array of monitored outputs (shape (4,) by default).
    """
    edges = jnp.asarray(genome, dtype=jnp.float32)
    if edges.ndim != 2 or edges.shape[1] < 3:
        raise ValueError("genome must be shape (E,3) with rows [from,to,weight]")

    from_idx = edges[:, 0].astype(jnp.int32)
    to_idx = edges[:, 1].astype(jnp.int32)
    weights = edges[:, 2].astype(jnp.float32)

    N = int(n_neurons)
    input_idxs = jnp.asarray(input_idxs, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float32)
    if data.shape[0] != input_idxs.shape[0]:
        raise ValueError("`data` length must match number of input_idxs")

    if output_idxs is None:
        output_idxs = jnp.arange(N - 5, N, dtype=jnp.int32)  # get last 5 outputs: [soma_x, soma_y, ax_x, ax_y, radius]
    else:
        output_idxs = jnp.asarray(output_idxs, dtype=jnp.int32)

    def step(state, _):
        # gather presynaptic values, multiply by weights, and accumulate to targets
        presyn = state[from_idx] * weights
        summed = jnp.zeros(N, dtype=jnp.float32).at[to_idx].add(presyn)
        # Use tanh + relu combination for more dynamic range
        new_state = jnp.tanh(summed) + jax.nn.relu(summed) * 0.1
        new_state = new_state.at[input_idxs].set(data)
        return new_state, None

    # initial state: zeros with inputs set
    state0 = jnp.zeros(N, dtype=jnp.float32).at[input_idxs].set(data)
    final_state, _ = jax.lax.scan(step, state0, None, length=n_steps)
    outputs = final_state[output_idxs]
    return outputs


def make_example_genome(n_neurons: int = 1000, n_connections: int = 5000, seed: int = 0):
    """Create an example feedforward genome with diverse connectivity patterns.
    Creates multiple processing layers and adds specific connection motifs
    to encourage spatial diversity in the neural layout.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 8)  # need several random keys
    
    # Create layered structure (virtual layers for organization)
    n_inputs = 2  # where external input goes
    n_hidden1 = n_neurons // 4
    n_hidden2 = n_neurons // 4
    n_outputs = 5  # last 5 neurons are outputs
    
    edges = []
    
    # 1. Connect inputs to first hidden layer with diverse weights
    for i in range(n_inputs):
        # Fan out from each input to first quarter of neurons
        targets = jax.random.randint(keys[0], (n_hidden1 // 2,), 2, n_hidden1)
        weights = jax.random.normal(keys[1], (len(targets),)) * 2.0  # stronger weights
        edges.extend([[i, t, w] for t, w in zip(targets, weights)])

    # 2. Connect first hidden layer to second with various motifs
    sources = jnp.arange(2, n_hidden1)
    for i in sources:
        # Create both local and long-range connections
        local = jax.random.randint(keys[2], (5,), i+1, min(i+20, n_hidden1 + n_hidden2))
        distant = jax.random.randint(keys[3], (5,), n_hidden1, n_hidden1 + n_hidden2)
        targets = jnp.concatenate([local, distant])
        weights = jax.random.normal(keys[4], (len(targets),))
        edges.extend([[i, t, w] for t, w in zip(targets, weights)])
    
    # 3. Add some skip connections from inputs to second hidden layer
    targets = jax.random.randint(keys[5], (n_hidden2 // 4,), n_hidden1, n_hidden1 + n_hidden2)
    weights = jax.random.normal(keys[6], (len(targets),)) * 1.5
    edges.extend([[0, t, w] for t, w in zip(targets, weights)])
    edges.extend([[1, t, w] for t, w in zip(targets, -weights)])  # opposing influence
    
    # 4. Connect second hidden to outputs with strong weights
    sources = jnp.arange(n_hidden1, n_hidden1 + n_hidden2)
    for i in sources:
        targets = jax.random.randint(keys[7], (3,), n_neurons - n_outputs, n_neurons)
        weights = jax.random.normal(keys[7], (len(targets),)) * 3.0  # strong output weights
        edges.extend([[i, t, w] for t, w in zip(targets, weights)])
    
    # Convert to array and normalize
    edges_arr = jnp.array(edges, dtype=jnp.float32)
    return normalize_genome(edges_arr, n_neurons=n_neurons, n_connections=n_connections, seed=seed)
    


if __name__ == "__main__":
    # quick example that builds an agent using a standardized feedforward genome
    example_genome = make_example_genome(n_neurons=1000, n_connections=5000, seed=42)
    agent = develop_agent("test", example_genome, n_neurons=1000, net_n_neurons=1000, net_steps=10)
    print("agent neurons shape:", agent["neurons"].shape)
    # print first 5 neurons' [soma_x, soma_y, ax_x, ax_y]
    print(agent["neurons"][:5])
