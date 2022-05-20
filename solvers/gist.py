from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from numba import njit

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
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

@njit
def mcp_value(w, lmbd, gamma):
    """Compute the value of MCP."""
    s0 = np.abs(w) < gamma * lmbd
    value = np.full_like(w, lmbd*gamma ** 2 / 2.)
    value[s0] = lmbd * np.abs(w[s0]) - w[s0]**2 / (2 *gamma)
    return np.sum(value)


def current_cost(X, y, w, lmbd, gamma):
    """Compute the following objective function.

    min_w 0.5 || y - X w||_2^2 +  \sum_k pen(|w_k|),

    where pen is then nonconvex regularizer.
    """
    normres2 = np.linalg.norm(y - X.dot(w))**2
    penalty = mcp_value(w,lmbd,gamma)
    return 0.5 * normres2/X.shape[0] + penalty



class Solver(BaseSolver):
    name = "GIST"  # proximal gradient, optionally accelerated

    requirements = ["numba"]
    references = [
        ' Gong et al."A General Iterative Shrinkage'
        ' and Thresholding Algorithm for Non-convex Regularized Optimization Problems"' 
        "ICML (2013)"
    ]

    # any parameter defined here is accessible as a class attribute
    tmax = 1e10
    eta = 1.5
    sigma = 0.1
    tol = 1e-6

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y, self.lmbd, self.gamma = X, y, lmbd, gamma

        # cache compilation time for prox_mcp_vec:
        self.run(1)

    def run(self, n_iter):
        n_samples, n_features = self.X.shape


        w = np.zeros(n_features)



        not_converged = False

        coutnew = current_cost(self.X, self.y, w, self.lmbd, self.gamma)
        wp_aux =  np.zeros(n_features,dtype=np.float64)
        for i in range(n_iter):
            t = 1
            #----------------------- gradient update  ------------------------
            grad = - self.X.T.dot(self.y - self.X.dot(w))/ n_samples
 
            wp_aux = prox_mcp_vec(w - grad/t,self.lmbd/t,self.gamma*t)
            
            #----------------------- backtracking stepsize ------------------------
            coutold = coutnew
            coutnew = current_cost(self.X, self.y, wp_aux, self.lmbd, self.gamma)
            while coutnew - coutold > - self.sigma / 2 * t * np.linalg.norm(w - wp_aux)**2:
                t = t * self.eta
                wp_aux = prox_mcp_vec(w - grad/t,self.lmbd/t,self.gamma*t)


                coutnew = current_cost(self.X, self.y, wp_aux,self.lmbd,self.gamma)

                if t > self.tmax:
                    print('Not converging. steps size too small!', t, self.tol)
                    not_converged = True
                    break
            w = wp_aux.copy()
            if not_converged:
                break



        self.w = w

    def get_result(self):
        return self.w
