from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from numba import njit

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


def prox_mcp_vec(x, lmbd, gamma):
    # because of gamma scaling by lipschitz, gamma can be < 1
    # in which case prox=IHT
    prox = x.copy()
    if gamma <= 1:
        prox[np.abs(x) <= lmbd] = 0.
    else:
        idx = np.abs(x) <= gamma * lmbd
        prox[idx] = gamma / (gamma - 1) * np.sign(x[idx]) * \
            np.maximum(0, np.abs(x[idx]) - lmbd)
    return prox


def mcp_value(w, lmbd, gamma):
    """Compute the value of MCP."""
    s0 = np.abs(w) < gamma * lmbd
    value = np.full_like(w, lmbd*gamma ** 2 / 2.)
    value[s0] = lmbd * np.abs(w[s0]) - w[s0]**2 / (2 * gamma)
    return np.sum(value)


def cost(X, y, w, lmbd, gamma):
    """Compute MCP Regression objective."""
    penalty = mcp_value(w, lmbd, gamma)
    return 0.5 * norm(y - X @ w)**2 / X.shape[0] + penalty


class Solver(BaseSolver):
    name = "GIST"  # proximal gradient, optionally accelerated

    requirements = ["numba"]
    references = [
        ' Gong et al."A General Iterative Shrinkage and Thresholding '
        'Algorithm for Non-convex Regularized Optimization Problems", '
        "ICML (2013)"
    ]

    # any parameter defined here is accessible as a class attribute
    eta = 1.5
    sigma = 0.001
    tol = 1e-6

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y, self.lmbd, self.gamma = X, y, lmbd, gamma

    def run(self, n_iter):
        n_samples, n_features = self.X.shape
        L = np.linalg.norm(self.X, ord=2) ** 2 / n_samples

        w = np.zeros(n_features)
        cost_new = np.mean(self.y**2)/2
        for k in range(n_iter):
            # TODO implement a proper BB rule
            # gradient update
            t = L

            grad = self.X.T @ (self.X @ w - self.y) / n_samples

            wp_aux = prox_mcp_vec(w - grad/t, self.lmbd/t, self.gamma*t)

            # backtracking stepsize
            cost_old = cost_new
            cost_new = cost(self.X, self.y, wp_aux, self.lmbd, self.gamma)

            while (cost_new - cost_old >
                    - self.sigma / 2 * t * norm(w - wp_aux)**2):
                t = t * self.eta
                wp_aux = prox_mcp_vec(w - grad/t, self.lmbd/t, self.gamma*t)
                cost_new = cost(self.X, self.y, wp_aux, self.lmbd, self.gamma)

                if t > L:
                    t = L
                    wp_aux = prox_mcp_vec(w - grad/t, self.lmbd/t,
                                          self.gamma*t)
                    break
            w[:] = wp_aux

        self.w = w

    def get_result(self):
        return self.w
