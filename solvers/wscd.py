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


@njit
def norm_col_sparse(data, indptr):
    n_features = len(indptr) - 1
    norms = np.zeros(n_features)
    for j in range(n_features):
        tmp = 0
        for idx in range(indptr[j], indptr[j + 1]):
            tmp += data[idx] ** 2
        norms[j] = np.sqrt(tmp)
    return norms


@njit
def cd(X, y, lmbd, gamma, lipschitz, n_iter, winit, tol=1e-5):
    n_samples, n_features = X.shape
    R = np.copy(y)
    w = np.zeros(n_features)
    # TODO a proper warm start
    # w = np.copy(winit)
    all_feats = np.arange(n_features)
    for _ in range(n_iter):
        for j in np.arange(n_features):
            if lipschitz[j]:
                old = w[j]
                w[j] = prox_mcp(
                    w[j] + X[:, j] @ R / (lipschitz[j] * n_samples),
                    lmbd / lipschitz[j],
                    gamma * lipschitz[j],
                )
                diff = old - w[j]
                if diff != 0:
                    R += diff * X[:, j]
        grad = - X.T.dot(y - X.dot(w)) / n_samples
        subdiff_dist = subdiff_distance(w, grad, all_feats, lmbd, gamma)
        if np.max(subdiff_dist) < tol:
            return w

    return w


@njit
def subdiff_distance(w, grad, ws, lmbd, gamma):
    """Compute distance of negative gradient to the subdifferential at w."""
    subdiff_dist = np.zeros_like(grad)
    for idx, j in enumerate(ws):
        if w[j] == 0:
            # distance of -grad to alpha * [-1, 1]
            subdiff_dist[idx] = max(0, np.abs(grad[idx]) - lmbd)
        elif np.abs(w[j]) < lmbd * gamma:
            # distance of -grad_j to (alpha - abs(w[j])/gamma) * sign(w[j])
            subdiff_dist[idx] = np.abs(
                grad[idx] + lmbd * np.sign(w[j])
                - w[j] / gamma)
        else:
            # distance of grad to 0
            subdiff_dist[idx] = np.abs(grad[idx])
    return subdiff_dist


@njit
def sparse_cd(
        X_data, X_indices, X_indptr, y, lmbd, gamma, lipschitz, n_iter, winit,
        tol=1e-5):
    n_features = len(X_indptr) - 1
    n_samples = len(y)
    all_feats = np.arange(n_features)
    w = np.zeros(n_features)
    R = np.copy(y)
    for _ in range(n_iter):
        for j in range(n_features):
            if lipschitz[j]:
                old = w[j]
                XjtR = 0.0
                for ind in range(X_indptr[j], X_indptr[j + 1]):
                    XjtR += X_data[ind] * R[X_indices[ind]]

                w[j] = prox_mcp(w[j] + XjtR / (lipschitz[j] * n_samples),
                                lmbd / lipschitz[j], gamma * lipschitz[j])
                diff = old - w[j]
                if diff != 0:
                    for ind in range(X_indptr[j], X_indptr[j + 1]):
                        R[X_indices[ind]] += diff * X_data[ind]
        subdiff_dist = subdiff_distance(w, R, all_feats, lmbd, gamma)
        if np.max(subdiff_dist) < tol:
            return w
    return w


@njit
def wscd(X, y, lmbd, gamma, lipschitz, n_iter, n_iter_outer, pruning=True,
         tol=1e-12, sparsity=False):

    n_samples, n_features = X.shape
    nb_feat_init = 10
    nb_feat_2_add = 30
    ind = np.argsort(-np.abs(X.T.dot(y)))[:nb_feat_init]

    all_feats = np.arange(n_features)
    # this is the initialization for the working set value
    w_init = np.zeros((nb_feat_init))
    # initialization of the full vector. use for computed the optimality
    w = np.zeros((n_features))

    for i in range(n_iter_outer):
        Xaux = X[:, ind]

        if sparsity:
            pass
            # TODO a proper call to the appropriate sparse function
        else:
            lip = lipschitz[ind]
            w_inter = cd(Xaux, y, lmbd, gamma, lip, n_iter, w_init, tol=tol)
        # pruning
        if pruning:
            nnz = (w_inter != 0)
            w_inter = w_inter[nnz]
            ind = ind[nnz]
            Xaux = Xaux[:, nnz]

        res = y - Xaux @ w_inter
        grad = - X.T @ res / n_samples
        w[ind] = w_inter
        subdiff_dist = subdiff_distance(w, grad, all_feats, lmbd, gamma)
        if np.max(np.abs(subdiff_dist)) < tol:
            return w
        else:
            w[ind] = 0
        # remove current working set from the candidate selection
        grad[ind] = 0
        # compute candidate
        candidate = np.argsort(-np.abs(grad))
        # TODO use argpartition when numba implems is available

        ind = np.hstack((ind, candidate[:nb_feat_2_add]))
        w_init = np.hstack((w_inter, np.zeros(nb_feat_2_add)))
    w = np.zeros(n_features)
    w[ind] = w_init

    return w


class Solver(BaseSolver):
    name = "WorkSet_CD"
    install_cmd = "conda"
    requirements = ["numba"]

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = X, y
        self.lmbd, self.gamma = lmbd, gamma
        self.n_iter_inner = 20000
        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        X = self.X
        if sparse.issparse(X):
            lipschitz = norm_col_sparse(X.data, X.indptr) ** 2 / len(self.y)
            sparsity = True
        else:
            sparsity = False
            lipschitz = np.sum(self.X ** 2, axis=0) / len(self.y)
        self.w = wscd(
            self.X, self.y, self.lmbd, self.gamma, lipschitz,
            self.n_iter_inner, n_iter, sparsity=sparsity)

    def get_result(self):
        return self.w
