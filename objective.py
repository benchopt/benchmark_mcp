import numpy as np
from benchopt.base import BaseObjective


class Objective(BaseObjective):
    name = "MCP Regression"

    parameters = {
        'reg': [0.1, 0.5],
        'gamma': [1.2, 1.2]
    }

    def __init__(self, reg=.1, gamma=1.2):
        self.reg = reg
        self.gamma = gamma

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()

    def compute(self, beta):
        diff = self.y - self.X.dot(beta)
        pen = (self.lmbd ** 2 * self.gamma / 2.) * np.ones(beta.shape)
        idx = np.abs(beta) <= self.gamma * self.lmbd
        gamma2 = self.gamma * 2
        pen[idx] = self.lmbd * np.abs(beta[idx]) - beta[idx] ** 2 / gamma2

        return diff.dot(diff) / 2. + pen.sum()

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd, gamma=self.gamma)
