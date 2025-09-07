import mpsnn_sim as mp
import jax
import jax.numpy as jnp
agents={}
def make_agent(genome,inputs=10,outputs=10):
    agent={"age":0,"task_age":0,"fitness":0,"inputs":inputs}
    agent["state"]=jnp.zeros(100)
    agent["spikes"] = jnp.zeros(50, dtype=bool)
    agent['genome']=genome
    agent["main"]=mp.develop_agent(genome,100) #value tbd
    agents[genome.tobytes()]=agent
    return agent
def run_agent(agent,inputs):
    agent["task_age"]+=1
    agent["state"],agent["spikes"],agent["refractory"]=mp.snn_step(agent["state"],agent["spikes"],jnp.zeros(100),agent["main"],inputs,dt=agent["task_age"])