import numpy as np
from benchopt import BaseObjective


def subdiff_distance(w, grad, lmbd, gamma):
    subdiff_dist = np.zeros_like(grad)
    for j in range(len(w)):
        if w[j] == 0:
            # distance of grad to alpha * [-1, 1]
            subdiff_dist[j] = max(0, np.abs(grad[j]) - lmbd)
        elif np.abs(w[j]) < lmbd * gamma:
            # distance of grad_j to (alpha - abs(w[j])/gamma) * sign(w[j])
            subdiff_dist[j] = np.abs(np.abs(grad[j]) - (
                lmbd - np.abs(w[j])/gamma))
        else:
            # distance of grad to 0
            subdiff_dist[j] = np.abs(grad[j])
    return subdiff_dist


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

        # compute distance of -grad f to subdifferential of MCP penalty
        grad = self.X.T @ diff / len(self.y)
        opt = subdiff_distance(beta, grad, self.lmbd, self.gamma)

        return dict(value=0.5 * diff @ diff + pen.sum(),
                    sparsity=(beta != 0).sum(), opt=opt.max())

    def _get_lambda_max(self):
        return abs(self.X.T @ self.y).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd, gamma=self.gamma)
