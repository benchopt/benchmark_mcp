from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features": [
            (100, 200),
        ],
        "normalize": [True, False],
    }

    def __init__(self, n_samples=10, n_features=50, normalize=False):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.normalize = normalize
        self.random_state = 0

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)
        if self.normalize:
            X /= (X ** 2).sum(axis=0)
        data = dict(X=X, y=y)

        return self.n_features, data
