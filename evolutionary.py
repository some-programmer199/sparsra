import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Any
from S_ops import S_op, make_s_op
import mpsnn_sim as mp
from cmaes import CMA

# =====================================================
# Genome Flattening/Unflattening Utilities
# =====================================================

def flatten_genome(genome):
    """Flatten a genome into a 1D array"""
    flat_parts = []
    dimensions = {}
    for key, value in genome.items():
        if isinstance(value, jnp.ndarray):
            flat_parts.append(value.flatten())
            dimensions[key] = value.shape
    return jnp.concatenate(flat_parts), dimensions

def unflatten_genome(flat_genome, dimensions):
    """Convert a flat array back into a genome dictionary"""
    genome = {}
    idx = 0
    for key, shape in dimensions.items():
        size = np.prod(shape)
        genome[key] = flat_genome[idx:idx+size].reshape(shape)
        idx += size
    return genome

# =====================================================
# Steady-State Evolution
# =====================================================

def evolve_s_ops(s_ops: List[S_op], fitness_func, iterations=10, population_size=None):
    """Evolve a population of S-ops using CMA-ES
    
    Args:
        s_ops: Initial population of S-ops
        fitness_func: Function to evaluate fitness of an S-op
        iterations: Number of evolution iterations
        population_size: Size of population for CMA-ES (optional)
        
    Returns:
        Evolved population of S-ops
    """
    if not s_ops:
        return []
    
    # Get the first S-op's genome and flatten it for CMA-ES
    flat_genome, dimensions = flatten_genome(s_ops[0].genome)
    
    # Initialize CMA-ES optimizer
    if population_size is None:
        population_size = 4 + int(3 * np.log(len(flat_genome)))
    optimizer = CMA(mean=flat_genome.tolist(), sigma=0.1, population_size=population_size)
    
    # Best solution tracking
    best_fitness = -float('inf')
    best_genome = None
    
    for i in range(iterations):
        # Sample new solutions
        solutions = []
        for _ in range(optimizer.population_size):
            solutions.append(optimizer.ask())
        
        # Create new S-ops from solutions
        new_s_ops = []
        fitnesses = []
        
        for solution in solutions:
            # Convert solution to genome
            solution_array = jnp.array(solution)
            genome = unflatten_genome(solution_array, dimensions)
            
            # Create S-op from genome
            s_op = make_s_op(genome)
            new_s_ops.append(s_op)
            
            # Evaluate fitness
            fitness = fitness_func(s_op)
            fitnesses.append(fitness)
            
            # Track best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome
        
        # Update CMA-ES (CMA-ES minimizes by default, so negate fitness)
        optimizer.tell(solutions, [-f for f in fitnesses])
        
        # Replace worst S-ops with best new ones
        all_s_ops = s_ops + new_s_ops
        all_fitnesses = [fitness_func(s_op) for s_op in s_ops] + fitnesses
        
        # Sort by fitness (descending)
        idx = jnp.argsort(-jnp.array(all_fitnesses))
        
        # Keep top population_size individuals
        s_ops = [all_s_ops[int(i)] for i in idx[:len(s_ops)]]
    
    return s_ops

# =====================================================
# Example
# =====================================================

if __name__ == "__main__":
    # Create a fitness function
    def fitness_function(s_op):
        # In a real scenario, this would evaluate how well the S-op performs
        # For this example, we'll just return a random value
        return np.random.random()
    
    # Create initial S-ops
    initial_s_ops = [
        make_s_op(weights=jnp.ones((10, 5)), input_size=10, output_size=5),
        make_s_op(weights=jnp.ones((5, 3)), input_size=5, output_size=3),
    ]
    
    # Evolve S-ops
    evolved_s_ops = evolve_s_ops(initial_s_ops, fitness_function, iterations=5, population_size=4)
    
    # Print results
    print(f"Evolved {len(evolved_s_ops)} S-ops")
    for i, s_op in enumerate(evolved_s_ops):
        print(f"S-op {i}: input_size={s_op.input_size}, output_size={s_op.output_size}")
        print(f"Weights shape: {s_op.weights.shape}")
        print()