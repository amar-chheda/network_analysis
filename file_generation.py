from landscapegenerator import landscapeGenerator
from graphgenerator import graphGenerator
import numpy as np
import networkx as nx
import  pickle
import os
from funcy import chunks

models_array = []
for i in range(40):
    if i < 10:
        models_array.append(landscapeGenerator(15,2))
    elif i >= 10 and i < 20:
        models_array.append(landscapeGenerator(15,4))
    elif i >= 20 and i < 30:
        models_array.append(landscapeGenerator(15,8))
    else:
        models_array.append(landscapeGenerator(15,12))

with open('./data/nk_landscape.pkl', 'wb') as f:
    pickle.dump(models_array,f)


n_agents = [24,48,60]
core = [(1/2),(1/3)]
pro = [0.7,0.8]

for n in n_agents:
    for c in core:
        for p in pro:
            graph = graphGenerator(n, int(c*n), p)
            with open(f'./data/agent_{n}_core_{int(c*n)}_prob_{int(p*100)}.pkl', 'wb') as f:
                pickle.dump(graph, f)


graph = graphGenerator(60,30,0.9)
with open(f'./data/agent_60_core_30_prob_90.pkl', 'wb') as f:
    pickle.dump(graph, f)

graph = graphGenerator(60,20,0.9)
with open(f'./data/agent_60_core_20_prob_90.pkl', 'wb') as f:
    pickle.dump(graph, f)



pexp = [0.6,0.7,0.8]
pconf = [0.2,0.3,0.4]
d = ['fixed', 'variable']
prisk = [0,0.05,0.1,0.5]

fnames = []
for name in os.listdir('./data'):
    print(name, name.startswith("agent"))
    if name.startswith("agent"):
        fnames.append(name)

inputs = []
for n in fnames:
    for di in d:
        for r in prisk:
            for pex in pexp:
                for pc in pconf:
                        inputs.append((n,di,r,pex,pc))

def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

x = list(divide_chunks(inputs, 8))
with open('./data/list_of_iterations.pkl', 'wb') as f:
    pickle.dump(x,f)
