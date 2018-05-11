import numpy as np


class AllData():
    """
    Class to create synthetic data and target
    """
    def __init__(self, n_samples, n_features):
        self.A = np.random.randn(n_samples, n_features)
        self.w = np.random.randn(n_features)
        self.b = np.sign(self.A.dot(self.w) + np.random.randn(n_samples))


