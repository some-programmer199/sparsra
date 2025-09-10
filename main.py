import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict

import mpsnn_sim as mp
from S_ops import S_op, make_s_op, apply_s_op, match_spike_sequence
from Agents import Agent, make_agent, run_agent, run_agent_vmapped
from evolutionary import evolve_s_ops
from generator import gen_problem

# =====================================================
# Configuration
# =====================================================
NUM_AGENTS = 5
NUM_S_OPS = 10
NUM_NEURONS = 100
NUM_OUTPUTS = 10
NUM_STEPS = 20
NUM_EVOLUTION_ITERATIONS = 5

# =====================================================
# S-op Fitness Evaluation
# =====================================================
def evaluate_s_op_fitness(s_op: S_op, agents: List[Agent], problem_inputs):
    """Evaluate the fitness of an S-op based on how well it helps agents solve a problem
    
    Args:
        s_op: The S-op to evaluate
        agents: List of agents to use for evaluation
        problem_inputs: Inputs representing the problem
        
    Returns:
        Fitness score for the S-op
    """
    # Create a copy of agents with this S-op as the only available one
    test_agents = []
    for agent in agents:
        test_agent = Agent(
            state=agent.state,
            spikes=agent.spikes,
            refractory=agent.refractory,
            task_age=agent.task_age,
            main=agent.main,
            threshold=agent.threshold,
            current_s_op=None,
            available_s_ops=[s_op],
            visited_s_ops=[]
        )
        test_agents.append(test_agent)
    
    # Stack agent fields for vmap
    agents_batched = Agent(
        state=jnp.stack([a.state for a in test_agents]),
        spikes=jnp.stack([a.spikes for a in test_agents]),
        refractory=jnp.stack([a.refractory for a in test_agents]),
        task_age=jnp.array([a.task_age for a in test_agents]),
        main=[a.main for a in test_agents],
        threshold=jnp.stack([a.threshold for a in test_agents]),
        current_s_op=[a.current_s_op for a in test_agents],
        available_s_ops=[a.available_s_ops for a in test_agents],
        visited_s_ops=[a.visited_s_ops for a in test_agents]
    )
    
    # Create S-op registry
    s_op_registry = {}
    
    # Run simulation for a few steps
    current_agents = agents_batched
    total_output = 0
    
    for step in range(NUM_STEPS):
        # Create batched inputs (in a real scenario, these would be problem-specific)
        inputs = jnp.ones((len(test_agents), NUM_NEURONS)) * 0.1
        inputs = inputs.at[:, 0].set(problem_inputs[step % len(problem_inputs)])
        
        # Run agents
        outputs, current_agents = run_agent_vmapped(current_agents, inputs, jnp.arange(NUM_OUTPUTS), s_op_registry)
        
        # Accumulate outputs (in a real scenario, this would be a task-specific reward)
        total_output += jnp.sum(outputs)
    
    # Calculate fitness based on total output
    # In a real scenario, this would be based on how well the agents solved the problem
    fitness = float(total_output)
    
    # Update the S-op's fitness
    s_op.fitness = fitness
    
    return fitness

# =====================================================
# Main Simulation Loop
# =====================================================
def solve_problem(problem, num_iterations=NUM_EVOLUTION_ITERATIONS):
    """Solve a problem using agents with S-ops in a steady-state evolutionary environment
    
    Args:
        problem: The problem to solve (ARC-like format)
        num_iterations: Number of evolutionary iterations
        
    Returns:
        Final set of agents and S-ops
    """
    key = jax.random.PRNGKey(42)
    
    # Extract input grid from the problem
    if isinstance(problem, dict) and 'train' in problem:
        # New ARC-like format
        input_grid = problem['train'][0]['input']
    else:
        # Old format
        input_grid = problem
    
    # Convert grid to flat array for SNN input
    problem_inputs = jnp.array(input_grid).flatten()
    
    # Create initial S-ops
    s_ops = []
    for i in range(NUM_S_OPS):
        s_op_key = jax.random.fold_in(key, i)
        s_op_genome = mp.init_cppn_params(s_op_key)
        s_op = make_s_op(s_op_genome)
        s_ops.append(s_op)
    
    # Create agents
    agents = []
    for i in range(NUM_AGENTS):
        agent_key = jax.random.fold_in(key, i + 100)
        agent_genome = mp.init_cppn_params(agent_key)
        agent = make_agent(agent_genome, num_neurons=NUM_NEURONS, available_s_ops=s_ops)
        agents.append(agent)
    
    # Main evolutionary loop
    for iteration in range(num_iterations):
        print(f"\nEvolutionary Iteration {iteration+1}/{num_iterations}")
        
        # Evaluate fitness of each S-op
        for i, s_op in enumerate(s_ops):
            fitness = evaluate_s_op_fitness(s_op, agents, problem_inputs)
            print(f"S-op {i} fitness: {fitness:.2f}")
        
        # Evolve S-ops using CMA-ES
        s_ops = evolve_s_ops(s_ops, lambda s_op: evaluate_s_op_fitness(s_op, agents, problem_inputs))
        
        # Update agents with new S-ops
        for agent in agents:
            agent.available_s_ops = s_ops
    
    return agents, s_ops

# =====================================================
# Example
# =====================================================

if __name__ == "__main__":
    # Generate a problem
    problem = gen_problem()
    
    # Solve the problem
    agents, evolved_s_ops = solve_problem(problem)
    
    # Print results
    print(f"\nFinal Results:")
    print(f"Number of agents: {len(agents)}")
    print(f"Number of evolved S-ops: {len(evolved_s_ops)}")
    
    # Print S-op fitness scores
    print("\nS-op Fitness Scores:")
    for i, s_op in enumerate(evolved_s_ops):
        print(f"S-op {i}: {s_op.fitness:.2f}")
    
    # Print agent statistics
    print("\nAgent S-op Usage:")
    for i, agent in enumerate(agents):
        if agent.current_s_op:
            print(f"Agent {i} is using an S-op with fitness {agent.current_s_op.fitness:.2f}")
        else:
            print(f"Agent {i} is not using any S-op")
        print(f"Agent {i} has visited {len(agent.visited_s_ops)} S-ops")
    
    print("\nSimulation complete.")