"""
This object takes in the values of n - number of nodes, m - number of nodes in the core and pStar - the probability of link formation between two nodes in core.
It returns a core preiphery structure network graph which represents the organizationall structure we want to study.
"""

#importing dependency
import  networkx as nx
import numpy as np

#initialize the class
class graphGenerator:

    def __init__(self, n, m, pStar):
        self.n = n
        self.m = m
        self.p = pStar
        self.core, self.peri = self.core_peri()
        self.g = self.core_peri_graph()

    def core_peri(self):
    """
    This function takes in the  n and m and gives out a list of core and periphery nodes
    """
        index = list(range(self.n))
        core = np.random.choice(index, size = self.m, replace = False)
        peri = list(set(index) - set(core))

        return core, peri

    def core_peri_graph(self):
    """
    This function takes in the core and peri nodes list from the core_peri() func and the value of pStar and generates a connected graph    
    """
        core = self.core
        peri = self.peri
        rand_mat = np.zeros((self.n, self.n))
        p1 = self.p
        p2 = 1-self.p
        ticker = False
        while ticker == False:
            for row in range(self.n):
                for col in range(self.n):
                    if row == col :
                        rand_mat[row][col] = 1

                    elif row != col and row in core and col in core and np.random.rand() <= p1:
                        rand_mat[row][col] = 1

                    elif row != col and row in core and col in peri and np.random.rand() <= p2:
                        rand_mat[row][col] = 1
            G = nx.from_numpy_array(rand_mat)
            ticker = nx.is_connected(G)

        return G
