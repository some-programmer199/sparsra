import jax
import jax.numpy as jnp
import mpsnn_sim as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, List

# =====================================================
# S-op Definition
# =====================================================
@dataclass
class S_op:
    # S-ops are small SNNs that transform inputs
    genome: Any  # CPPN genome parameters
    snn: Dict = None  # Developed SNN network
    num_neurons: int = 50  # Size of the SNN
    fitness: float = 0.0  # Fitness score for evolution

# =====================================================
# S-op Creation
# =====================================================
def make_s_op(genome: Any, num_neurons: int = 50) -> S_op:
    """Create an S-op with a small SNN developed from the genome"""
    snn = mp.develop_agent(genome, num_neurons)
    return S_op(genome=genome, snn=snn, num_neurons=num_neurons)

# =====================================================
# S-op Execution
# =====================================================
def apply_s_op(s_op: S_op, input_data, state=None, spikes=None, refractory=None):
    """Apply an S-op to input data, handling variable input sizes
    
    Args:
        s_op: The S-op to apply
        input_data: Input data of any size
        state, spikes, refractory: Optional SNN state from previous step
        
    Returns:
        new_state, new_spikes, new_refractory: Updated SNN state
    """
    # Initialize SNN state if not provided
    if state is None:
        state = jnp.zeros(s_op.num_neurons)
    if spikes is None:
        spikes = jnp.zeros(s_op.num_neurons, dtype=bool)
    if refractory is None:
        refractory = jnp.zeros(s_op.num_neurons, dtype=int)
    
    # Handle input size mismatch
    input_size = input_data.size
    target_size = s_op.num_neurons
    
    # Pad or truncate input to match SNN size
    if input_size < target_size:
        # Pad with zeros or repeat the input
        repetitions = target_size // input_size + 1
        padded_input = jnp.tile(input_data, repetitions)[:target_size]
        inputs = padded_input
    elif input_size > target_size:
        # Truncate the input
        inputs = input_data[:target_size]
    else:
        inputs = input_data
    
    # Run one step of the SNN
    new_state, new_spikes, new_refractory = mp.snn_step(
        state, spikes, refractory, s_op.snn, inputs
    )
    
    return new_state, new_spikes, new_refractory

# =====================================================
# S-op Selection
# =====================================================
def match_spike_sequence(spike_sequence: jnp.ndarray, s_ops: List[S_op]) -> int:
    """Select an S-op based on the given spike sequence
    
    Args:
        spike_sequence: The spike sequence from the agent
        s_ops: List of available S-ops
        
    Returns:
        Index of the selected S-op
    """
    if not s_ops:
        return -1
    
    # For now, we'll use a simple selection method based on the spike count
    # This will be replaced with a more sophisticated tag-based system later
    spike_count = jnp.sum(spike_sequence)
    
    # Select S-op based on spike count modulo number of S-ops
    # This ensures a deterministic but varied selection
    if spike_count > 0:
        return int(spike_count % len(s_ops))
    else:
        # If no spikes, select the first S-op
        return 0

# =====================================================
# S-op Combination
# =====================================================
def combine_s_ops(s_ops: list[S_op], key) -> S_op:
    """Create a new S-op by combining multiple S-ops
    
    This creates a new genome that represents the combined behavior
    of the input S-ops. The new S-op will have its own SNN.
    """
    # For now, we'll just use the first S-op's genome as a placeholder
    # In a more advanced implementation, this would merge the genomes
    if len(s_ops) == 0:
        return None
    
    # In a real implementation, we would combine the genomes in a meaningful way
    # For now, we'll just use the first S-op's genome
    combined_genome = s_ops[0].genome
    
    # Create a new S-op with the combined genome
    return make_s_op(combined_genome)

# =====================================================
# Example S-ops
# =====================================================
if __name__ == "__main__":
    # Create a random genome for an S-op
    key = jax.random.PRNGKey(42)
    genome = mp.init_cppn_params(key)
    
    # Create an S-op with the genome
    s_op = make_s_op(genome, num_neurons=50)
    
    # Create some input data
    input_data = jnp.zeros(20)
    input_data = input_data.at[:5].set(10.0)  # Activate first 5 neurons
    
    # Apply the S-op to the input data
    state, spikes, refractory = apply_s_op(s_op, input_data)
    
    # Print the results
    print(f"Input shape: {input_data.shape}")
    print(f"State shape: {state.shape}")
    print(f"Number of spikes: {spikes.sum()}")
    
    # Create another S-op
    key2 = jax.random.PRNGKey(43)
    genome2 = mp.init_cppn_params(key2)
    s_op2 = make_s_op(genome2, num_neurons=30)
    
    # Combine the S-ops
    combined_op = combine_s_ops([s_op, s_op2], key)
    
    # Apply the combined S-op to the input data
    state2, spikes2, refractory2 = apply_s_op(combined_op, input_data)
    
    print(f"\nCombined S-op results:")
    print(f"State shape: {state2.shape}")
    print(f"Number of spikes: {spikes2.sum()}")
    
    # Test with different input sizes
    large_input = jnp.ones(100)
    state3, spikes3, refractory3 = apply_s_op(s_op, large_input)
    
    print(f"\nLarge input results:")
    print(f"Input shape: {large_input.shape}")
    print(f"State shape: {state3.shape}")
    print(f"Number of spikes: {spikes3.sum()}")
    
    print("\nS-ops created and applied successfully.")