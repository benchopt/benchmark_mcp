from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from scipy.sparse import issparse
    import numpy as np
    from scipy.linalg import norm


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': [
            "bodyfat", "leukemia", "news20.binary", "rcv1.binary", "finance",
            "real-sim"],
    }

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm(self.dataset)

        if issparse(self.X):
            self.X.multiply(
                np.sqrt(len(self.y)) / np.sqrt(self.X.power(2).sum(axis=0)))
        else:
            self.X = np.array(self.X)
            self.X /= np.linalg.norm(self.X, axis=0)
            self.X *= np.sqrt(len(self.y))

        data = dict(X=self.X, y=self.y)

        return self.X.shape[1], data
