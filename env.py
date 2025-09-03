import Agents
import cmaes
population_size=50
evolver=cmaes.CMA(mean=[0.0]*10,sigma=0.5,popsize=population_size)
agents0=[evolver.ask() for _ in range(population_size)]
def run_evolution(agents, generations=100):
    for gen in range(generations):
        pass
    
    