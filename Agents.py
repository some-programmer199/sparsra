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
def run_agent(agent,inputs,output_idxs=None):
    agent["task_age"]+=1
    agent["state"],agent["spikes"],agent["refractory"]=mp.snn_step(agent["state"],agent["spikes"],jnp.zeros(100),agent["main"],inputs,dt=agent["task_age"])
    #returns the state at neurons corresponding to output idxs
    return agent["state"][output_idxs]
run_agent_vmapped=jax.vmap(run_agent,in_axes=(None,0))
if __name__=="__main__":
    agent=make_agent(jnp.array([[0,1,0.5],[1,2,0.3],[2,3,0.8]]),inputs=10,outputs=10)
    inputs=jnp.zeros((5,100))
    inputs=inputs.at[:,0].set(10.0)
    output=run_agent_vmapped(agent,inputs,output_idxs=jnp.arange(10))
    print(output)