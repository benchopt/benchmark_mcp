import numpy as np

from benchopt.base import BaseSolver
from benchopt.util import safe_import_context


with safe_import_context() as import_ctx:
    from scipy import sparse
    from numba import njit


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def prox_mcp_vec(x, lmbd, gamma):
    st_abs = np.maximum(np.abs(x) - lmbd, 0)
    return np.sign(x) * np.minimum(np.abs(x), gamma / (gamma - 1) * st_abs)


class Solver(BaseSolver):
    name = 'Python-PGD'  # proximal gradient, optionally accelerated

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y, self.lmbd, self.gamma = X, y, lmbd, gamma

    def run(self, n_iter):
        if sparse.issparse(self.X):
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        else:
            L = np.linalg.norm(self.X, ord=2) ** 2

        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        if self.use_acceleration:
            z = np.zeros(n_features)

        t_new = 1
        for _ in range(n_iter):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / L
                w = prox_mcp_vec(z, self.lmbd / L, self.gamma)
                z = w + (t_old - 1.) / t_new * (w - w_old)
            else:
                w -= self.X.T @ (self.X @ w - self.y) / L
                w = prox_mcp_vec(w, self.lmbd / L, self.gamma)

        self.w = w

    def get_result(self):
        return self.w
