from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from numba import njit


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def st(x, mu):
    if x > mu:
        return x - mu
    if x < -mu:
        return x + mu
    return 0


@njit
def prox_mcp(x, lmbd, gamma):
    if gamma < 1:
        return 0 if abs(x) <= lmbd else x
        # TODO potential numerical errors in <=  leading to hard discontinuity

    if x > gamma * lmbd:
        return x
    if x < -gamma * lmbd:
        return x
    return gamma / (gamma - 1) * st(x, lmbd)


class Solver(BaseSolver):
    name = "cd"
    install_cmd = "conda"
    requirements = ["numba"]
    references = [
        'P. Breheny and J. Huang, "Coordinate descent algorithms for '
        "nonconvex  penalized regression, with applications to biological "
        'feature selection for Scaling Sparse Optimization", '
        "Ann. Appl. Stat.,  vol. 5, pp. 232 (2011)"
    ]

    def set_objective(self, X, y, lmbd, gamma):
        # use Fortran order to compute gradient on contiguous columns
        self.X, self.y = np.asfortranarray(X), y
        self.lmbd, self.gamma = lmbd, gamma

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        if sparse.issparse(self.X):
            # TODO this only works when the lipschitz constants are 1
            self.w = self.sparse_cd(
                self.X.data,
                self.X.indices,
                self.X.indptr,
                self.y,
                self.lmbd,
                self.gamma,
                n_iter,
            )
        else:
            lipschitz = np.sum(self.X ** 2, axis=0) / len(self.y)
            self.w = self.cd(
                self.X, self.y, self.lmbd, self.gamma, lipschitz, n_iter
            )

    @staticmethod
    @njit
    def cd(X, y, lmbd, gamma, lipschitz, n_iter):
        n_samples, n_features = X.shape
        R = np.copy(y)
        w = np.zeros(n_features)
        for _ in range(n_iter):
            for j in range(n_features):
                old = w[j]
                w[j] = prox_mcp(
                    w[j] + X[:, j] @ R / (lipschitz[j] * n_samples),
                    lmbd / lipschitz[j],
                    gamma * lipschitz[j],
                )
                diff = old - w[j]
                if diff != 0:
                    R += diff * X[:, j]
        return w

    @staticmethod
    @njit
    def sparse_cd(X_data, X_indices, X_indptr, y, lmbd, gamma, n_iter):
        n_features = len(X_indptr) - 1
        n_samples = len(y)
        w = np.zeros(n_features)
        R = np.copy(y)
        for _ in range(n_iter):
            for j in range(n_features):
                old = w[j]
                grad = 0.0
                for ind in range(X_indptr[j], X_indptr[j + 1]):
                    grad += X_data[ind] * R[X_indices[ind]]

                w[j] = prox_mcp(w[j] + grad / n_samples, lmbd, gamma)
                diff = old - w[j]
                if diff != 0:
                    for ind in range(X_indptr[j], X_indptr[j + 1]):
                        R[X_indices[ind]] += diff * X_data[ind]
        return w

    def get_result(self):
        return self.w
