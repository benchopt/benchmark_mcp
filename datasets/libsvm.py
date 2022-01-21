from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from scipy.sparse import issparse
    import numpy as np
    from scipy.linalg import norm
    from sklearn.preprocessing import normalize


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': [
            "bodyfat", "leukemia", "news20.binary", "rcv1.binary", "finance",
            "real-sim"]
    }

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self, dataset="bodyfat"):
        # import ipdb; ipdb.set_trace()
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm(self.dataset)

        normalize(self.X, axis=0, copy=False)
        self.X *= np.sqrt(len(self.y))

        data = dict(X=self.X, y=self.y)

        return self.X.shape[1], data
