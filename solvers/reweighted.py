from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from flashcd.estimators import WeightedLasso


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def gprime(x, lmbd, gamma):
    return np.maximum(
        0, lmbd - np.abs(x) / gamma)


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
        # use Fortran order to compute gradient on contiguous columns
        self.X, self.y = np.asfortranarray(X), y
        self.lmbd, self.gamma = lmbd, gamma

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        # how to set n_iter for benchopt, on outer iterations or inner ?
        self.w = self.reweighted(self.X, self.y, self.lmbd, self.gamma,
                                 n_iter=n_iter, n_iter_weighted=5)

    @staticmethod
    def reweighted(X, y, lmbd, gamma, n_iter, n_iter_weighted=5):
        # First weights is equivalent to a simple Lasso
        weights = np.ones(X.shape[1])
        for k in range(n_iter_weighted):
            clf = WeightedLasso(alpha=lmbd / len(y), tol=1e-12,
                                fit_intercept=False,
                                weights=weights, max_iter=n_iter)
            clf.fit(X, y)
            coefs = clf.coef_
            # Update weights
            weights = gprime(coefs, lmbd / len(y), gamma * len(y))
        return coefs

    def get_result(self):
        return self.w
