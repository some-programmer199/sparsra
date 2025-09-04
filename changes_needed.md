**needed changes:**

##### Agents.py

* develop_agent must return a dict of :{"id":agentid,"genome" genome,"age":0,"fitness":0,"main connections":edge_index(in syntax [from,to],"main neuron weights":a array of 4 values per neuron,"main neuron spike":a array of 4 values per neuron,"state":a bunch of zeroes)
* run_genome takes genome and neuron idx, and outputs soma x and y, axon x and y, a 4 value set of weights, and a 4 value spike
* run_agent will run an agent for one timestep and take a agent dict and external inputs(an array that is added to the agents neurons voltage totals), it uses snn_sim.py's simulation function to do dynamics, it will return an updated agent dict

##### snn_sim.py

* sim_mpsnn takes connections and state and updates the state
