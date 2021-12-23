import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "MCP Regression"

    parameters = {"reg": [1, 0.5, 0.1, 0.01], "gamma": [3]}

    def __init__(self, reg=0.1, gamma=1.2):
        self.reg = reg
        self.gamma = gamma

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()

    def compute(self, beta):
        diff = self.y - self.X @ beta
        pen = np.full(len(beta), 0.5 * self.lmbd ** 2 * self.gamma)
        idx = np.abs(beta) <= self.gamma * self.lmbd
        pen[idx] = (self.lmbd * np.abs(beta[idx]) -
                    0.5 * beta[idx] ** 2 / self.gamma)

        return dict(value=0.5 * diff @ diff / len(diff) + pen.sum(),
                    sparsity=(beta != 0).sum())

    def _get_lambda_max(self):
        return abs(self.X.T @ self.y).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd, gamma=self.gamma)
