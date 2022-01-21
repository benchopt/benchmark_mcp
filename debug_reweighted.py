import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from benchopt.datasets import make_correlated_data
from flashcd import WeightedLasso


def deriv_mcp(x, lmbd, gamma):
    return np.maximum(0, lmbd - np.abs(x) / gamma)


X, y, _ = make_correlated_data(n_samples=100, n_features=200, random_state=0)
X /= np.linalg.norm(X, axis=0) / np.sqrt(len(y))


alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
lmbd = 0.0001 * alpha_max
gamma = 20


# def reweighted(X, y, lmbd, gamma, n_iter, n_reweightings=11):
#     # First weights is equivalent to a simple Lasso
#     if n_iter == 0:
#         return np.zeros(X.shape[1])

#     weights = lmbd * np.ones(X.shape[1])
#     clf = WeightedLasso(alpha=1, tol=1e-12,
#                         fit_intercept=False,
#                         weights=weights, max_iter=n_iter,
#                         warm_start=True, verbose=1)
#     for _ in range(n_reweightings):
#         clf.penalty.weights = weights
#         old_weights = weights.copy()
#         clf.fit(X, y)
#         # Update weights as derivative of MCP penalty
#         weights = deriv_mcp(clf.coef_, lmbd, gamma)
#     return clf.coef_


def mcp_val(w):
    pen = lmbd ** 2 * gamma / 2. * np.ones(w.shape[0])
    idx = np.abs(w) <= lmbd * gamma
    pen[idx] = lmbd * np.abs(w[idx]) - w[idx] ** 2 / (2 * gamma)
    return pen.sum()


# E = []
# for n_iter in range(1000,1001):
#     w = reweighted(X, y, lmbd, gamma, n_iter)
#     R = X @ w - y
#     obj = norm(R) ** 2 / (2 * len(R)) + mcp_val(w)
#     E.append(obj)

# E = np.array(E)
# plt.semilogy(E - np.min(E))
# plt.show(block=False)


def subdiff_distance(w, grad, lmbd, gamma):
    """Distance of negative gradient to Fréchet subdifferential of MCP at w."""
    subdiff_dist = np.zeros_like(grad)
    for j in range(len(w)):
        if w[j] == 0:
            # distance of grad to [-lmbd, lmbd]
            subdiff_dist[j] = max(0, np.abs(grad[j]) - lmbd)
        elif np.abs(w[j]) < lmbd * gamma:
            # distance of -grad_j to (lmbd - abs(w[j])/gamma) * sign(w[j])
            subdiff_dist[j] = np.abs(
                grad[j] + lmbd * np.sign(w[j]) - w[j] / gamma)
        else:
            # distance of grad to 0
            subdiff_dist[j] = np.abs(grad[j])
    return subdiff_dist


# w = reweighted(X, y, lmbd, gamma, n_iter=1000)*
n_iter = 1000
weights = lmbd * np.ones(X.shape[1])
clf = WeightedLasso(alpha=1, tol=1e-12,
                    fit_intercept=False,
                    weights=weights, max_iter=n_iter,
                    warm_start=True, verbose=0)
for _ in range(50):
    clf.penalty.weights = weights
    old_weights = weights.copy()
    clf.fit(X, y)
    # Update weights as derivative of MCP penalty
    print(weights[125])
    weights = deriv_mcp(clf.coef_, lmbd, gamma)

w = clf.coef_
R = X @ w - y

grad = X.T @ (X @ w - y) / len(y)

subdist = subdiff_distance(w, grad, lmbd, gamma)


idx = np.where(np.logical_and(w != 0, np.abs(w) <= gamma * lmbd))
print(idx)
print(subdist[idx])
