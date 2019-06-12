from landscapegenerator import landscapeGenerator
from graphgenerator import graphGenerator
import pickle

model_array = []
for i in range(30):
    if i <10:
        model_array.append(landscapeGenerator(15,2))
    elif i >= 10 and i <20:
        model_array.append(landscapeGenerator(15,4))
    else:
        model_array.append(landscapeGenerator(15,8))

with open('./data/nk_landscape.pkl', 'wb') as f:
    pickle.dump(model_array,f)

graph = graphGenerator(40,15,0.7)

with open('./data/graph_40_nodes.pkl', 'wb') as f:
    pickle.dump(graph,f)                   
