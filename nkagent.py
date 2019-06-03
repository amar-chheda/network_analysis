import numpy as np

class nkAgent:

    def __init__(self, model, iniSol, assign, risk=0.5, dist=1):
        self.n = len(iniSol)
        self.risk = risk
        self.model = model
        self.sol = iniSol
        self.score = model.cal_fit(self.sol)
        self.d = dist
        self.assign = assign

    def mutate(self, alpha=1):
        tmp_sol = self.sol.copy()
        index = list(range(self.n))
        mut_ind = np.random.choice(index, size = self.d)

        for i in mut_ind:
            if tmp_sol[i] == 1:
                tmp_sol[i] = 0
            else:
                tmp_sol[i] = 1

        tmp_score = self.model.cal_fit(tmp_sol)
        if tmp_score > self.score or np.random.rand() < self.risk * alpha:
            self.sol = tmp_sol
            self.score = tmp_score
