import mpsnn_sim as mp
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any

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

# =====================================================
# Agent creation
# =====================================================
def make_agent(genome, num_neurons=100, num_spikes=50):
    state = jnp.zeros(num_neurons)
    spikes = jnp.zeros(num_spikes, dtype=bool)
    refractory = jnp.zeros(num_neurons, dtype=int)
    task_age = jnp.array(0)
    main = mp.develop_agent(genome, num_neurons)
    threshold = main["threshold"]
    return Agent(state, spikes, refractory, task_age, main, threshold)

# =====================================================
# Run single agent step (functional)
# =====================================================
def run_agent(agent: Agent, inputs, output_idxs=None):
    # increment task_age
    task_age = agent.task_age + 1

    # step SNN
    new_state, new_spikes, new_refractory = mp.snn_step(
        agent.state,
        agent.spikes,
        agent.refractory,
        agent.main,
        inputs,
        dt=task_age
    )

    # return updated agent and output neurons
    new_agent = Agent(
        state=new_state,
        spikes=new_spikes,
        refractory=new_refractory,
        task_age=task_age,
        main=agent.main,
        threshold=agent.threshold
    )

    if output_idxs is None:
        return new_state, new_agent
    else:
        return new_state[output_idxs], new_agent

# =====================================================
# VMAP helpers
# =====================================================
# in_axes=(0,0) → batch over both agents and inputs
run_agent_vmapped = jax.vmap(run_agent, in_axes=(0, 0, None))

# =====================================================
# Example
# =====================================================
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    genome = mp.init_cppn_params(key)  # example genome
    num_agents = 3
    num_inputs = 100
    num_outputs = 10

    # create batched agents
    agents_list = [make_agent(genome, num_neurons=num_inputs) for _ in range(num_agents)]
    
    # stack agent fields for vmap
    agents_batched = Agent(
        state=jnp.stack([a.state for a in agents_list]),
        spikes=jnp.stack([a.spikes for a in agents_list]),
        refractory=jnp.stack([a.refractory for a in agents_list]),
        task_age=jnp.array([a.task_age for a in agents_list]),
        main=[a.main for a in agents_list],  # list is fine if same across agents
        threshold=jnp.stack([a.threshold for a in agents_list])
    )

    # batched inputs
    inputs = jnp.zeros((num_agents, num_inputs))
    inputs = inputs.at[:, 0].set(10.0)

    # run agents
    outputs, new_agents = run_agent_vmapped(agents_batched, inputs, jnp.arange(num_outputs))
    print("Outputs shape:", outputs.shape)
