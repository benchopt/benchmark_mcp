from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from skglm import MCPRegression
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip:skglm'
    ]
    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel'
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = X, y
        self.lmbd, self.gamma = lmbd, gamma

        self.model = MCPRegression(
            alpha=lmbd, gamma=gamma, fit_intercept=False, tol=0)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.run(1)  # Make sure we cache the numba compilation.

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1]])
            return

        self.model.max_iter = n_iter
        self.model.fit(self.X, self.y)

        self.coef = self.model.coef_.flatten()

    def get_result(self):
        return self.coef
