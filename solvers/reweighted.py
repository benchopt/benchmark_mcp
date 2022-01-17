from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from flashcd.estimators import WeightedLasso
    # from flashcd.penalties import WeightedL1
    # from scipy.sparse import issparse

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def deriv_mcp(x, lmbd, gamma):
    return np.maximum(0, lmbd - np.abs(x) / gamma)


class Solver(BaseSolver):
    name = "reweighted"
    install_cmd = "conda"
    requirements = ["numba"]
    references = [
        'E. J. Candès, M. B. Wakin, S. P. Boyd, '
        "Enhancing Sparsity by Reweighted l1 Minimization"
        "Journal of Fourier Analysis and Applications,"
        "vol. 14, pp. 877-905 (2008)"
    ]

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = X, y
        self.lmbd, self.gamma = lmbd, gamma

        # cache the numba compilation.
        self.run(2)

    def run(self, n_iter):
        # how to set n_iter for benchopt, on outer iterations or inner ?
        self.w = self.reweighted(self.X, self.y, self.lmbd, self.gamma,
                                 n_iter=n_iter, n_iter_weighted=40)

    @staticmethod
    def reweighted(X, y, lmbd, gamma, n_iter, n_iter_weighted):
        # First weights is equivalent to a simple Lasso
        weights = lmbd * np.ones(X.shape[1])
        clf = WeightedLasso(alpha=1, tol=1e-12,
                            fit_intercept=False,
                            weights=weights, max_iter=n_iter,
                            warm_start=True)
        for _ in range(n_iter_weighted):
            clf.penalty.weights = weights
            clf.fit(X, y)
            # Update weights as derivative of MCP penalty
            weights = deriv_mcp(clf.coef_, lmbd, gamma)
        return clf.coef_

    def get_result(self):
        return self.w
