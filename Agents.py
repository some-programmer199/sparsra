import mpsnn_sim as mp
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from S_ops import S_op, apply_s_op, match_spike_sequence

# =====================================================
# Agent Dataclass
# =====================================================
@dataclass
class Agent:
    state: jnp.ndarray
    spikes: jnp.ndarray
    refractory: jnp.ndarray
    task_age: jnp.ndarray
    main: Any  # CPPN-developed network dict
    threshold: jnp.ndarray  # optional, can be modified each step
    current_s_op: Optional[S_op] = None  # Current S-op the agent is using
    available_s_ops: List[S_op] = field(default_factory=list)  # Available S-ops
    visited_s_ops: List[S_op] = field(default_factory=list)  # S-ops already visited

# =====================================================
# Agent creation
# =====================================================
def make_agent(genome, num_neurons=100, num_spikes=50, available_s_ops=None):
    state = jnp.zeros(num_neurons)
    spikes = jnp.zeros(num_spikes, dtype=bool)
    refractory = jnp.zeros(num_neurons, dtype=int)
    task_age = jnp.array(0)
    main = mp.develop_agent(genome, num_neurons)
    threshold = main["threshold"]
    
    if available_s_ops is None:
        available_s_ops = []
        
    return Agent(
        state=state, 
        spikes=spikes, 
        refractory=refractory, 
        task_age=task_age, 
        main=main, 
        threshold=threshold,
        current_s_op=None,
        available_s_ops=available_s_ops,
        visited_s_ops=[]
    )

# =====================================================
# S-op Selection
# =====================================================
def select_s_op(agent: Agent, s_op_registry: Dict[str, S_op] = None):
    """Select an S-op based on the agent's spike sequence
    
    Args:
        agent: The agent selecting an S-op
        s_op_registry: Dictionary of available S-ops indexed by their ID
        
    Returns:
        Updated agent with selected S-op
    """
    if not agent.available_s_ops:
        return agent
    
    # Use the agent's spike sequence to match against S-op tags
    best_idx = match_spike_sequence(agent.spikes, agent.available_s_ops)
    
    if best_idx == -1:
        return agent  # No match found
    
    selected_s_op = agent.available_s_ops[best_idx]
    
    # Check if this S-op has already been chosen by another agent
    if s_op_registry is not None:
        s_op_id = id(selected_s_op)
        
        # If this S-op is already in the registry, use the existing one
        if str(s_op_id) in s_op_registry:
            selected_s_op = s_op_registry[str(s_op_id)]
        else:
            # Otherwise, add it to the registry
            s_op_registry[str(s_op_id)] = selected_s_op
    
    # Update the agent's current S-op and visited list
    visited_s_ops = agent.visited_s_ops + [selected_s_op]
    
    return Agent(
        state=agent.state,
        spikes=agent.spikes,
        refractory=agent.refractory,
        task_age=agent.task_age,
        main=agent.main,
        threshold=agent.threshold,
        current_s_op=selected_s_op,
        available_s_ops=agent.available_s_ops,
        visited_s_ops=visited_s_ops
    )

# =====================================================
# Run single agent step (functional)
# =====================================================
def run_agent(agent: Agent, inputs, output_idxs=None, s_op_registry=None):
    # increment task_age
    task_age = agent.task_age + 1
    
    # Apply current S-op if available
    if agent.current_s_op is not None:
        # Process inputs through the current S-op
        s_op_outputs = apply_s_op(agent.current_s_op, inputs)
        # Combine original inputs with S-op outputs
        # Here we're using the S-op outputs as additional inputs
        combined_inputs = jnp.concatenate([inputs, s_op_outputs])
        # Truncate to match expected input size if needed
        if combined_inputs.shape[0] > inputs.shape[0]:
            combined_inputs = combined_inputs[:inputs.shape[0]]
    else:
        combined_inputs = inputs

    # step SNN
    new_state, new_spikes, new_refractory = mp.snn_step(
        agent.state,
        agent.spikes,
        agent.refractory,
        agent.main,
        combined_inputs,
        dt=task_age
    )

    # Create updated agent
    new_agent = Agent(
        state=new_state,
        spikes=new_spikes,
        refractory=new_refractory,
        task_age=task_age,
        main=agent.main,
        threshold=agent.threshold,
        current_s_op=agent.current_s_op,
        available_s_ops=agent.available_s_ops,
        visited_s_ops=agent.visited_s_ops
    )
    
    # Select a new S-op based on the agent's spike pattern
    new_agent = select_s_op(new_agent, s_op_registry)

    if output_idxs is None:
        return new_state, new_agent
    else:
        return new_state[output_idxs], new_agent

# =====================================================
# VMAP helpers
# =====================================================
# in_axes=(0,0,None,None) → batch over both agents and inputs, shared output_idxs and s_op_registry
run_agent_vmapped = jax.vmap(run_agent, in_axes=(0, 0, None, None))

# =====================================================
# Example
# =====================================================
if __name__ == "__main__":
    from S_ops import make_s_op
    
    key = jax.random.PRNGKey(0)
    num_agents = 3
    num_inputs = 100
    num_outputs = 10
    num_s_ops = 5
    
    # Create a set of S-ops
    s_ops = []
    for i in range(num_s_ops):
        s_op_key = jax.random.fold_in(key, i)
        s_op_genome = mp.init_cppn_params(s_op_key)
        s_op = make_s_op(s_op_genome)
        s_ops.append(s_op)
    
    # Create S-op registry (for tracking shared S-ops)
    s_op_registry = {}
    
    # Create agent genomes
    agent_genomes = []
    for i in range(num_agents):
        agent_key = jax.random.fold_in(key, i + 100)  # Different from S-op keys
        agent_genome = mp.init_cppn_params(agent_key)
        agent_genomes.append(agent_genome)
    
    # Create agents with available S-ops
    agents_list = [make_agent(genome, num_neurons=num_inputs, available_s_ops=s_ops) 
                  for genome in agent_genomes]
    
    # Stack agent fields for vmap
    agents_batched = Agent(
        state=jnp.stack([a.state for a in agents_list]),
        spikes=jnp.stack([a.spikes for a in agents_list]),
        refractory=jnp.stack([a.refractory for a in agents_list]),
        task_age=jnp.array([a.task_age for a in agents_list]),
        main=[a.main for a in agents_list],
        threshold=jnp.stack([a.threshold for a in agents_list]),
        current_s_op=[a.current_s_op for a in agents_list],
        available_s_ops=[a.available_s_ops for a in agents_list],
        visited_s_ops=[a.visited_s_ops for a in agents_list]
    )

    # Batched inputs
    inputs = jnp.zeros((num_agents, num_inputs))
    inputs = inputs.at[:, 0].set(10.0)

    # Run agents with S-op selection
    outputs, new_agents = run_agent_vmapped(agents_batched, inputs, jnp.arange(num_outputs), s_op_registry)
    print("Outputs shape:", outputs.shape)
    
    # Check which S-ops were selected
    for i, agent in enumerate(new_agents):
        if agent.current_s_op is not None:
            print(f"Agent {i} selected an S-op")
        else:
            print(f"Agent {i} did not select an S-op")
    
    # Run for multiple steps to see S-op selection behavior
    current_agents = new_agents
    for step in range(5):
        outputs, current_agents = run_agent_vmapped(current_agents, inputs, jnp.arange(num_outputs), s_op_registry)
        
        # Print visited S-ops count for each agent
        for i, agent in enumerate(current_agents):
            print(f"Step {step+1}, Agent {i}: {len(agent.visited_s_ops)} S-ops visited")
    
    print("\nS-op registry size:", len(s_op_registry))
    print("Simulation complete")
