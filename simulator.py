#importing all the dependencies
import numpy as np
from nkagent import nkAgent #this is a localy designed object
import pickle
from multiprocessing import Pool
import networkx as nx
import matplotlib.pyplot as plt
import time
import os



class Simulator(nkAgent):

    """
    This object takes in the nkAgent object and the values for number of iterations, time_step conditions and returns the
    result for the simulation.
    """
    #initialized the object
    def __init__(self, folder_name, p_exp, iterations, time_step):

        with open(f'./{folder_name}/graph_40_nodes.pkl', 'rb') as f:
            self.graph = pickle.load(f)  # class attribute

        # loads the stored NK landscape
        with open(f'./{folder_name}/nk_landscape.pkl', 'rb') as f:
            self.model_array = pickle.load(f)  # class attribute
        self.G = self.graph.g #connected graph
        self.iterations = iterations #Number of iterations to be performed for each combination
        self.core = self.graph.core #indexes of core nodes
        self.peri = self.graph.peri #indexes of periphery indexes
        self.n_agents = self.graph.n #number of total agents (Number of nodes in graph)
        self.p_connect = self.graph.p #probability of connection within the core nodes
        self.risk = 0 #probability of risk taking
        self.dist = 1 #Distance of exploration
        self.time_step = time_step #Max number of time steps to search the space
        self.assignments = ['exp_core', 'exp_peri', 'random'] #Assignments to agents for exploration
        self.p_exp = p_exp #Probability of exploration
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
    def node_update(G, j, alpha, core, peri, p):
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

        for i in neigh:
            score_list.append([i, G.node[i]['agent'].score])
        max_neigh = max(score_list, key=lambda x: x[1])

        if assign == 'exp_core':

            if G.node[j]['agent'].score >= max_neigh[1] and j in core and np.random.rand() <= p:
                G.node[j]['agent'].mutate(alpha)
            elif G.node[j]['agent'].score < max_neigh[1]:
                G.node[j]['agent'].sol = G.node[max_neigh[0]]['agent'].sol
                G.node[j]['agent'].score = G.node[max_neigh[0]]['agent'].score

        elif assign == 'exp_peri':

            if G.node[j]['agent'].score >= max_neigh[1] and j in peri and np.random.rand() <= p:
                G.node[j]['agent'].mutate(alpha)
            elif G.node[j]['agent'].score < max_neigh[1]:
                G.node[j]['agent'].sol = G.node[max_neigh[0]]['agent'].sol
                G.node[j]['agent'].score = G.node[max_neigh[0]]['agent'].score

        elif assign == 'random':

            if G.node[j]['agent'].score >= max_neigh[1] and np.random.rand() <= p:
                G.node[j]['agent'].mutate(alpha)
            elif G.node[j]['agent'].score < max_neigh[1]:
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
            self.G = self.node_update(self.G, i, alpha, self.core, self.peri, self.p_exp)
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
                        iniSol = np.random.randint(2, size=model.n)
                        agent_array.append(nkAgent(model= model, iniSol=iniSol, risk=self.risk, assign=ass, dist= self.dist))
                    self.G = self.graph_init(G= self.G, agent = agent_array)
                    alpha = 1
                    s = []
                    for j in range(self.time_step):
                        score = self.network_update(alpha)
                        s.append(score)
                        alpha = 0.95 * alpha
                    results[f'model_{m}']['assignments'][f'assignment_{a}']['iterations'][f'iter_{i}'][
                        'agents'] = agent_array
                    results[f'model_{m}']['assignments'][f'assignment_{a}']['iterations'][f'iter_{i}']['score'] = s

                a += 1
            m += 1
        return results


def plot_all(result, title):
    fig, ax = plt.subplots(1, 3, sharex='row', figsize=(18, 5))
    s_model = []
    x = 10
    for i in range(len(result.keys())):
        s_ass = []
        mmax = result[f'model_{i}']['max']
        mmin = result[f'model_{i}']['min']
        for a in range(3):
            s_it = []
            for it in range(len(result[f'model_{i}']['assignments'][f'assignment_{a}']['iterations'].keys())):
                score = result[f'model_{i}']['assignments'][f'assignment_{a}']['iterations'][f'iter_{it}']['score']
                # s_it.append(score)
                # s_it.append([((sc-mmin)/(mmax-mmin)) for sc in score])
                s_it.append([((1 / np.exp(x)) * (np.exp(x * sc) - 1)) for sc in score])
            s_ass.append(np.mean(np.array(s_it), axis=0))
        s_model.append(np.array(s_ass))
    ax[0].plot(np.mean(np.array(s_model)[:10], axis=0).T, linewidth=1)
    ax[1].plot(np.mean(np.array(s_model)[10:20], axis=0).T, linewidth=1)
    ax[2].plot(np.mean(np.array(s_model)[20:], axis=0).T, linewidth=1)
    ax[0].set_title('Complexity low, k = 2')
    ax[1].set_title('Complexity med, k = 4')
    ax[2].set_title('Complexity high, k = 8')
    fig.suptitle(title)
    plt.legend(['exp_core', 'exp_peri', 'random'])
    plt.savefig(f'./data/{title}.png')




if __name__ == '__main__':
    start = time.time()
    p = Pool(os.cpu_count())
    res = p.starmap(Simulator, [('data', 0.5, 50,100),('data', 0.6, 50,100),('data', 0.7, 50,100),('data', 0.8, 50,100),('data', 0.9, 50,100)])
    i = 0
    for r in res:
        with open(f'./data/multiprocess_results_{i}', 'wb') as f:
            pickle.dump(r.result,f)
        i+=1
    stop = time.time()
    
    print(f'Total time taken for the process to finish is: {(stop-start)/60} Minutes')
