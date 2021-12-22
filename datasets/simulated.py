from benchopt import BaseDataset, safe_import_context
from benchopt.datasets.simulated import make_correlated_data

with safe_import_context():
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (100, 500),
        ],
        'rho': [0.5],
        'normalize': [True],
    }

    def __init__(
            self, n_samples=10, n_features=50, rho=0, normalize=True,
            random_state=27):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.rho = rho
        self.normalize = normalize

    def get_data(self):

        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, rho=self.rho,
            random_state=self.random_state
        )

        if self.normalize:
            X /= np.linalg.norm(X, axis=0)
        data = dict(X=X, y=y)
        return self.n_features, data
