from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import pycasso
    import numpy as np
    from numpy.linalg import norm


class Solver(BaseSolver):
    name = "pycasso"

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = X, y
        self.lmbd, self.gamma = lmbd, gamma
        lmbd_max = norm(X.T @ y, ord=np.inf) / len(y)
        self.clf = pycasso.Solver(
            X, y, lambdas=(1, lmbd / lmbd_max), penalty="mcp", gamma=gamma,
            useintercept=False, family="gaussian")

        self.clf.prec = 1e-12

    def run(self, n_iter):
        self.clf.max_ite = n_iter
        self.clf.train()

    def get_result(self):
        return self.clf.coef()['beta'].flatten()
