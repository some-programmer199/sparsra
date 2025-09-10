import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from S_ops import S_op, apply_s_op
from tree import TreeNode, add_child

# =====================================================
# Search Algorithm
# =====================================================

def select_node(root: TreeNode, exploration_weight: float = 1.0) -> TreeNode:
    """Select a node to expand using UCB1 formula."""
    # Start at root
    node = root
    
    # Traverse down the tree until we reach a leaf node
    while node.children:
        # If any child has not been visited, select it
        unvisited = [child for child in node.children if child.visits == 0]
        if unvisited:
            return unvisited[0]
        
        # Otherwise, select the child with the highest UCB score
        node = max(node.children, key=lambda c: ucb_score(c, exploration_weight))
    
    return node

def ucb_score(node: TreeNode, exploration_weight: float) -> float:
    """Calculate the UCB1 score for a node."""
    # If the node has not been visited, return infinity
    if node.visits == 0:
        return float('inf')
    
    # Calculate exploitation term
    exploitation = node.value / node.visits
    
    # Calculate exploration term
    parent_visits = node.parent.visits if node.parent else 1
    exploration = exploration_weight * jnp.sqrt(jnp.log(parent_visits) / node.visits)
    
    return exploitation + exploration

def expand(node: TreeNode, available_s_ops: List[S_op], input_data: jnp.ndarray) -> TreeNode:
    """Expand a node by adding a child with a new S-op."""
    # If there are no available S-ops, return the node
    if not available_s_ops:
        return node
    
    # Choose a random S-op from the available ones
    # In a more advanced implementation, we would use a heuristic
    s_op = available_s_ops[0]  # For simplicity, just take the first one
    
    # Apply the S-op to get the new state
    new_state, _, _ = apply_s_op(s_op, input_data)
    
    # Add a child node with the new S-op and state
    child = add_child(node, s_op, new_state)
    
    return child

def simulate(node: TreeNode, input_data: jnp.ndarray, reward_func) -> float:
    """Simulate from the node to estimate its value."""
    # For now, just use the reward function directly on the node's state
    return reward_func(node.state)

def backpropagate(node: TreeNode, value: float):
    """Update the node and its ancestors with the simulation result."""
    # Update the node
    node.visits += 1
    node.value += value
    
    # Update all ancestors
    current = node.parent
    while current:
        current.visits += 1
        current.value += value
        current = current.parent

def mcts_search(root: TreeNode, available_s_ops: List[S_op], input_data: jnp.ndarray, 
               reward_func, num_iterations: int = 100, exploration_weight: float = 1.0) -> TreeNode:
    """Run Monte Carlo Tree Search to find the best S-op sequence."""
    for _ in range(num_iterations):
        # Selection: select a node to expand
        node = select_node(root, exploration_weight)
        
        # Expansion: add a child node with a new S-op
        if node.visits > 0:  # Only expand nodes that have been visited
            node = expand(node, available_s_ops, input_data)
        
        # Simulation: simulate from the node to estimate its value
        value = simulate(node, input_data, reward_func)
        
        # Backpropagation: update the node and its ancestors
        backpropagate(node, value)
    
    # Return the best child of the root
    if not root.children:
        return None
    
    return max(root.children, key=lambda c: c.value / max(c.visits, 1))

# =====================================================
# Example
# =====================================================

if __name__ == "__main__":
    # This is a placeholder for a real example
    # In a real scenario, you would:
    # 1. Create a root node with an initial state
    # 2. Create a list of available S-ops
    # 3. Define a reward function
    # 4. Run MCTS to find the best S-op sequence
    print("Search algorithm defined.")