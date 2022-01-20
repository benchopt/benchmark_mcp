from benchopt import BaseDataset, safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.preprocessing import normalize


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features": [
            (100, 200),
        ],
        "scale": [False, True],
        "X_density": [0.5],
    }

    def __init__(
            self, n_samples=10, n_features=50, scale=False, X_density=1):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.scale = scale
        self.X_density = X_density
        self.random_state = 0

    def get_data(self):

        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, X_density=self.X_density,
            random_state=self.random_state)
        X[:, :X.shape[1] // 3] *= 10
        X[:, -X.shape[1] // 3:] /= 10
        if self.scale:
            normalize(X, axis=0, copy=False)
            X *= np.sqrt(len(y))
        data = dict(X=X, y=y)

        return self.n_features, data
