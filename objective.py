import numpy as np
from benchopt.base import BaseObjective


class Objective(BaseObjective):
    name = "MCP Regression"

    parameters = {
        'reg': [[.1, 1.2], [.5, 1.2]]  # [lbda ratio, gamma
    }

    def __init__(self, reg=[.1, 1.2], fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        print(self.reg)
        self.lmbd = self.reg[0] * self._get_lambda_max()
        self.gamma = self.reg[1]
    # def set_data(self, X, y):
    #     self.X, self.y = X, y
    #     print(self.reg)
    #     regzip = zip(*self.reg)
    #     lmbd, gamma = [list(tup) for tup in regzip]
    #     self.lmbd = lmbd * self._get_lambda_max()
    #     self.gamma = gamma

    def compute(self, beta):
        diff = self.y - self.X.dot(beta)
        # beta_norm = np.sqrt(norms2 / n_samples) * beta

        pen = (self.lmbd ** 2 * self.gamma / 2.) * np.ones(beta.shape)
        small_idx = np.abs(beta) <= self.gamma * self.lmbd
        pen[small_idx] = self.lmbd * np.abs(beta[small_idx]) - beta[small_idx] ** 2 / (2 * self.gamma)

        return diff.dot(diff) / 2. + pen.sum()

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, reg=self.reg)
        #           fit_intercept=self.fit_intercept)
