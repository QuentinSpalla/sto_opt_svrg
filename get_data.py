
import numpy as np


class AllData():
    """
    Data from file with methods adding financial indicators
    """
    def __init__(self, n_samples, n_features): #in_file):
        self.A = np.random.randn(n_samples, n_features)
        self.w = np.random.randn(n_features)
        self.b = np.sign(self.A.dot(self.w) + np.random.randn(n_samples))
        """
        self.df_prices = pd.read_csv(in_file, sep=';')
        self.data = self.df_prices.copy()
        self.df_target = None
        self.first_idx_ret = 0
        """

