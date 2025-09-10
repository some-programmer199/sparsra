# SPARSRA: Spiking Neural Networks for ARC Prize 2025

This project implements a framework for solving ARC (Abstraction and Reasoning Corpus) tasks using Spiking Neural Networks (SNNs) with a tree search approach.

## Project Structure

- `mpsnn_sim.py`: Core SNN simulator with LIF neurons and refractory periods
- `S_ops.py`: Implementation of S-ops (Spiking Operators) that transform inputs using small SNNs
- `tree.py`: Tree structure for representing search space of S-op combinations
- `search.py`: Monte Carlo Tree Search (MCTS) algorithm for finding optimal S-op sequences
- `Agents.py`: Agent implementation with SNN-based decision making
- `generator.py`: Generator for ARC-like problems
- `main.py`: Main entry point demonstrating how to use the framework

## Concept

The core idea is to use small Spiking Neural Networks (S-ops) as building blocks for solving ARC tasks. Each S-op is a small SNN that transforms input data in some way. By combining these S-ops in different ways, we can create complex transformations that solve ARC tasks.

Agents use Monte Carlo Tree Search to explore the space of possible S-op combinations, selecting the most promising ones to solve the task.

## Usage

To run the framework:

```bash
python main.py
```

This will generate a random problem, create some S-ops, and use MCTS to find a sequence of S-ops that solves the problem.

## Extending

To extend the framework:

1. Create new S-ops by defining new genomes
2. Implement a better reward function in `main.py`
3. Add more sophisticated problem generation in `generator.py`
4. Improve the MCTS algorithm in `search.py`

## Contributing

Contributions are welcome! Any help getting the framework running and improving it will be appreciated.