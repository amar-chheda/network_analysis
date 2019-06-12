#importing all the dependencies
import numpy as np
from nkagent import nkAgent #this is a localy designed object
import pickle
from multiprocessing import Pool
import networkx as nx
import time
import os



class Simulator(nkAgent):

    """
    This object takes in the nkAgent object and the values for number of iterations, time_step conditions and returns the
    result for the simulation.
    """
    #initialized the object
    def __init__(self, graph_name, dist, risk, p_exp, p_conf, red_fact = 0.95):

        with open(f'./data/{graph_name}', 'rb') as f:
            self.graph = pickle.load(f)  # class attribute
        # loads the stored NK landscape

        with open(f'./data/nk_landscape.pkl', 'rb') as f:
            self.model_array = pickle.load(f)  # class attribute

        self.G = self.graph.g #connected graph
        self.red_fact = red_fact #Reduction factor for risk in each time_step
        self.iterations = 100 #Number of iterations to be performed for each combination
        self.core = self.graph.core #indexes of core nodes
        self.peri = self.graph.peri #indexes of periphery indexes
        self.n_agents = self.graph.n #number of total agents (Number of nodes in graph)
        self.p_connect = self.graph.p #probability of connection within the core nodes
        self.risk = risk #probability of risk taking
        self.dist = dist #Distance of exploration
        self.time_step = 200 #Max number of time steps to search the space
        self.assignments = ['exp_core', 'exp_peri', 'random'] #Assignments to agents for exploration
        self.p_exp = p_exp #Probability of exploration
        self.p_conf = p_conf #probability of confirmation
        self.result = self.main_loop() #running the simulation to get the result

    @staticmethod
    def graph_init(G, agent):
        """
        This function initializes the graph with each agent in one node.
        :param G: Blank graph structure (NetworkX library)
        :param agent: list of agents
        :return: initialized graph
        """
        agent_dict = {}
        i = 0
        for agent in agent:
            agent_dict[i] = {'agent': agent}
            i += 1
        nx.set_node_attributes(G, agent_dict)
        return G

    @staticmethod
    def node_update(G, j, alpha, core, peri, p_exp, p_conf):
        """
        This function checks the assignment and updates the score according to the given conditions
        :param G: Initialized graph
        :param j: Index of node to be updated
        :param alpha: Reduction factor for the probability of risk
        :param core: Indexes for the core nodes
        :param peri: Indexes for the periphery nodes
        :param p: Probability of exploration
        :return: Graph with updated scores
        """
        assign = G.node[j]['agent'].assign
        neigh = G.neighbors(j)
        score_list = []
        is_exp = np.random.choice(['explore', 'confirm'], p = [p_exp, (1-p_exp)])
        is_conf = np.random.choice(['confirm', 'explore'], p = [p_conf, (1-p_conf)])

        for i in neigh:
            score_list.append([i, G.node[i]['agent'].score])
        max_neigh = max(score_list, key=lambda x: x[1])

        if assign == 'exp_core' and j in core and G.node[j]['agent'].score < max_neigh[1] :

            if is_exp == 'explore':
                G.node[j]['agent'].mutate(alpha)
            elif is_exp == 'confirm':
                G.node[j]['agent'].sol = G.node[max_neigh[0]]['agent'].sol
                G.node[j]['agent'].score = G.node[max_neigh[0]]['agent'].score

        elif assign == 'exp_peri' and j in peri and G.node[j]['agent'].score < max_neigh[1] :

            if is_exp == 'explore':
                G.node[j]['agent'].mutate(alpha)
            elif is_exp == 'confirm':
                G.node[j]['agent'].sol = G.node[max_neigh[0]]['agent'].sol
                G.node[j]['agent'].score = G.node[max_neigh[0]]['agent'].score

        elif assign == 'random' and G.node[j]['agent'].score < max_neigh[1]:

            if is_exp == 'explore':
                G.node[j]['agent'].mutate(alpha)
            elif is_exp == 'confirm':
                G.node[j]['agent'].sol = G.node[max_neigh[0]]['agent'].sol
                G.node[j]['agent'].score = G.node[max_neigh[0]]['agent'].score

        elif  G.node[j]['agent'].score < max_neigh[1]:

            if is_conf == 'explore':
                G.node[j]['agent'].mutate(alpha)
            elif is_conf == 'confirm':
                G.node[j]['agent'].sol = G.node[max_neigh[0]]['agent'].sol
                G.node[j]['agent'].score = G.node[max_neigh[0]]['agent'].score

        return G



    def network_update(self, alpha):
        """
        This function updates all the nodes in the graph within the given constraints
        :param alpha: Reduction factor for the probability of risk
        :return: graph with updated scores at the nodes
        """
        TS = 0
        for i in self.G.nodes():
            self.G = self.node_update(self.G, i, alpha, self.core, self.peri, self.p_exp, self.p_conf)
            TS += self.G.node[i]['agent'].score
        return TS / len(self.G)


    def main_loop(self):
        """
        This function contains main logic for the entire simulation run
        :return: returns the results of the simulation run in Json format
        """
        results = {}
        m = 0
        for model in self.model_array:
            results[f'model_{m}'] = {}
            results[f'model_{m}']['model'] = model
            results[f'model_{m}']['N'] = model.n
            results[f'model_{m}']['K'] = model.k
            mmax = model.max
            mmin = model.min
            results[f'model_{m}']['max'] = mmax
            results[f'model_{m}']['min'] = mmin
            results[f'model_{m}']['assignments'] = {}
            a = 0
            for ass in self.assignments:
                results[f'model_{m}']['assignments'][f'assignment_{a}'] = {}
                results[f'model_{m}']['assignments'][f'assignment_{a}']['ass'] = ass
                results[f'model_{m}']['assignments'][f'assignment_{a}']['iterations'] = {}
                for i in range(self.iterations):
                    results[f'model_{m}']['assignments'][f'assignment_{a}']['iterations'][f'iter_{i}'] = {}
                    agent_array = []
                    for agen in range(len(self.G)):
                        if self.dist == 'fixed':
                            d = 1
                        elif self.dist == 'variable':
                            d = np.random.choice([1,2,3], p = [0.25,0.5,0.25])
                        iniSol = np.random.randint(2, size=model.n)
                        agent_array.append(nkAgent(model= model, iniSol=iniSol, risk=self.risk, assign=ass, dist= d))
                    self.G = self.graph_init(G= self.G, agent = agent_array)
                    if self.risk == 0.5:
                        red = self.red_fact
                    else:
                        red = 1
                    alpha = 1
                    s = []
                    for j in range(self.time_step):
                        score = self.network_update(alpha)
                        s.append(score)
                        alpha = red * alpha
                    results[f'model_{m}']['assignments'][f'assignment_{a}']['iterations'][f'iter_{i}'][
                        'agents'] = agent_array
                    results[f'model_{m}']['assignments'][f'assignment_{a}']['iterations'][f'iter_{i}']['score'] = s

                a += 1
            m += 1
        return results



###########################################################################################################################################################

with open('./data/list_of_iterations.pkl', 'rb') as f:
    simulation_list = pickle.load(f)

if __name__ == '__main__':
    i = 0
    j = 0
    for sublist in simulation_list:
        print(time.asctime( time.localtime(time.time()) ))
        print('\n')
        print(f'Initiating {j} to {j+8} set of simulations.....')
        print('\n')
        print('The combinations being simulated are: \n')
        print(sublist)
        start = time.time()
        p = Pool(os.cpu_count())
        res = p.starmap(Simulator, sublist)
        for r in res:
            with open(f'./data/sim_results/run_list_of_iteration_{i}.pkl', 'wb') as f:
                pickle.dump(r.result,f)
            i+=1
        j+=8
        stop = time.time()

        print(f'Total time taken for the process to finish is: {(stop-start)/60} Minutes')
