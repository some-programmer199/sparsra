import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Callable

# =====================================================
# S-op Definition
# =====================================================
@dataclass
class S_op:
    # S-ops are functions that transform agent states or environments
    func: Callable
    # Parameters for the S-op, can be static or learned
    params: Any = None

# =====================================================
# S-op Creation
# =====================================================
def make_s_op(func: Callable, params: Any = None) -> S_op:
    return S_op(func=func, params=params)

# =====================================================
# S-op Execution
# =====================================================
def apply_s_op(agent, env, s_op: S_op):
    # An S-op can modify the agent, the environment, or both
    return s_op.func(agent, env, s_op.params)

# =====================================================
# S-op Combination
# =====================================================
def combine_s_ops(s_ops: list[S_op]) -> S_op:
    # Creates a new S-op that applies a sequence of S-ops
    def combined_func(agent, env, params_list):
        for i, s_op in enumerate(s_ops):
            agent, env = apply_s_op(agent, env, s_op)
        return agent, env
    
    # Pass an empty list for params, as individual s_ops already have theirs
    return make_s_op(combined_func, [])

# =====================================================
# Example S-ops (to be expanded)
# =====================================================
def example_s_op_func(agent, env, params):
    # This is a placeholder for a real S-op function
    # For example, it could modify the agent's state based on the environment
    print("Applying an example S-op")
    return agent, env

if __name__ == "__main__":
    # Example of creating and combining S-ops
    s_op1 = make_s_op(example_s_op_func)
    s_op2 = make_s_op(example_s_op_func)
    
    combined_op = combine_s_ops([s_op1, s_op2])
    
    # In a real scenario, you would apply this to an agent and environment
    # apply_s_op(some_agent, some_env, combined_op)
    print("S-ops created and combined successfully.")