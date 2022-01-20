import numpy as np
from benchopt import BaseObjective


def subdiff_distance(w, grad, lmbd, gamma):
    """Distance of negative gradient to Fréchet subdifferential of MCP at w."""
    subdiff_dist = np.zeros_like(grad)
    for j in range(len(w)):
        if w[j] == 0:
            # distance of grad to [-lmbd, lmbd]
            subdiff_dist[j] = max(0, np.abs(grad[j]) - lmbd)
        elif np.abs(w[j]) < lmbd * gamma:
            # distance of grad_j to (lmbd - abs(w[j])/gamma) * sign(w[j])
            subdiff_dist[j] = np.abs(
                -grad[j] + lmbd * np.sign(w[j]) - w[j] / gamma)
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
        pen = (self.lmbd ** 2 * self.gamma / 2.) * np.ones(beta.shape)
        idx = np.abs(beta) <= self.gamma * self.lmbd
        gamma2 = self.gamma * 2
        pen[idx] = self.lmbd * np.abs(beta[idx]) - beta[idx] ** 2 / gamma2

        # compute distance of -grad f to subdifferential of MCP penalty
        grad = self.X.T @ diff / len(self.y)
        opt = subdiff_distance(beta, grad, self.lmbd, self.gamma)

        return dict(value=0.5 * diff @ diff / len(self.y) + pen.sum(),
                    sparsity=(beta != 0).sum(), opt_violation=opt.max())

    def _get_lambda_max(self):
        return abs(self.X.T @ self.y).max() / len(self.y)

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd, gamma=self.gamma)
