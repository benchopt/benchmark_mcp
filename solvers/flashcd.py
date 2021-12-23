from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from flashcd import MCP
    import numpy as np


class Solver(BaseSolver):
    name = "flashcd"

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = np.asfortranarray(X), y
        self.lmbd, self.gamma = lmbd, gamma
        self.clf = MCP(alpha=lmbd, gamma=gamma,
                       fit_intercept=False)

        # Make sure we cache the numba compilation.
        self.run(1)
        self.clf.tol = 1e-12

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_
