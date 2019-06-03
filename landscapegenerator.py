import numpy as np
import itertools


class landscapeGenerator:

    def __init__(self, N, K):
        self.n = N
        self.k = K
        self.nk = self.nk_landscape()
        self.dep = self.dependency_structure()
        self.max = np.max(self.nk)
        self.min = np.min(self.nk)

    def disp_n(self):
        return print(f'Value of N is : {self.n}')

    def disp_k(self):
        return print(f'Value of K is : {self.k}')

    def dependency_structure(self):
        '''
        This function takes the number of elements and interdependency and
        creates a random interaction matrix with a diagonal filled.
        '''
        dep = {}
        for i in np.arange(self.n):
            indi = list(range(self.n))
            indi.remove(i)
            dep[i] = np.random.choice(indi, self.k, replace=False).tolist()

        return dep

    def nk_landscape(self):
        '''
        Generates an NK landscape - an array of random numbers ~U(0, 1).
        '''
        nk = np.random.rand(self.n,2**(self.k+1),self.n)

        return nk

    def cal_fit(self, bit_str):
        '''
            Takes landscape and a combination and returns a vector of fitness
            values (contribution value for each of for N decision variables)
        '''
        if len(bit_str) == self.n:

            score = 0
            ind_dict = list(itertools.product(range(2), repeat = self.k + 1))
            for i in range(len(bit_str)):
                dep_bit = self.dep[i].copy()
                dep_bit.append(i)
                inter_bits = tuple([bit_str[x] for x in dep_bit])
                ind = ind_dict.index(inter_bits)
                score += self.nk[i][ind][i]
            return score / self.n
        else:
            return print('The length of bit string should be equal to the length of N')