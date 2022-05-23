from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
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
def cd(X, y, lmbd, gamma, lipschitz, n_iter, winit=None, tol=1e-3):
    n_samples, n_features = X.shape
    R = np.copy(y)
    w = np.zeros(n_features)
    all_feats = np.arange(n_features)
    for _ in range(n_iter):
        for j in range(n_features):
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
def wscd(X, y, lmbd, gamma, lipschitz, n_iter, n_iter_outer, pruning=True,
         tol=1e-8):

    n_features = X.shape[1]
    nb_feat_init = 10
    nb_feat_2_add = 30
    ind = np.argsort(-np.abs(X.T.dot(y)))[:nb_feat_init]

    all_feats = np.arange(n_features)
    winit = np.zeros((nb_feat_init))
    w = np.zeros((n_features))

    for i in range(n_iter_outer):
        Xaux = X[:, ind]

        w_inter = cd(Xaux, y, lmbd, gamma, lipschitz, n_iter, winit)

        # pruning
        if pruning:
            w_inter = w_inter[w_inter.nonzero()[0]]
            ind = ind[w_inter.nonzero()[0]]
            Xaux = Xaux[:, w_inter.nonzero()[0]]

        res = y - Xaux @ w_inter
        grad = - X.T @ res

        subdiff_dist = subdiff_distance(w, grad, all_feats, lmbd, gamma)
        if np.max(subdiff_dist) < tol:
            return w

        candidate = np.argsort(-np.abs(grad))
        nb_add = 0
        for cand in candidate:
            if cand not in ind:
                ind = np.hstack((ind, np.array([cand])))
                nb_add += 1
                if nb_add == nb_feat_2_add:
                    break

        w_init = np.hstack((w_inter, np.zeros(nb_add)))

    w = np.zeros(n_features)
    w[ind] = w_init

    return w


class Solver(BaseSolver):
    name = "working set cd"
    install_cmd = "conda"
    requirements = ["numba"]

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = X, y
        self.lmbd, self.gamma = lmbd, gamma
        self.n_iter_outer = 10
        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        lipschitz = np.sum(self.X ** 2, axis=0) / len(self.y)
        self.w = wscd(
            self.X, self.y, self.lmbd, self.gamma, lipschitz, n_iter,
            self.n_iter_outer)

    @staticmethod
    @njit
    def sparse_cd(
            X_data, X_indices, X_indptr, y, lmbd, gamma, lipschitz, n_iter):
        n_features = len(X_indptr) - 1
        n_samples = len(y)
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
        return w

    def get_result(self):
        return self.w
