import numpy as np
from numba import njit
import gradient_tool


class SVRG():
    def __init__(self, data, target, nbr_epoch, T, learning_rate):
        self.data = data
        self.target = target
        self.nbr_epoch = nbr_epoch
        self.T = T
        self.learning_rate = learning_rate
        self.theta_k = np.zeros(data.shape[1])
        self.theta_bar = self.theta_k.copy()
        self.grad_history = np.ones(nbr_epoch)

    def train(self):
        t = 0
        grad_avg = gradient_tool.grad_jit(self.target, self.data, self.theta_bar)

        for i in range(self.nbr_epoch):
            # .. save gradient (for plotting purposes) every epoch ..
            if t == self.T-1:
                self.theta_bar = self.theta_k.copy()
                grad_avg = gradient_tool.grad_jit(self.target, self.data, self.theta_bar)
                self.grad_history[i] = np.linalg.norm(grad_avg)
                t = 0

            # .. pick random sample ..
            idx = np.random.randint(0, self.data.shape[0])

            # .. compute and apply SVRG update rule ..
            cur_grad = gradient_tool.partial_grad_jit(self.target, self.data, self.theta_k, idx)
            prev_grad = gradient_tool.partial_grad_jit(self.target, self.data, self.theta_bar, idx)
            svrg_update = cur_grad - prev_grad + grad_avg
            self.theta_k = self.theta_k - self.learning_rate * svrg_update
            t += 1


class CM():
    def __init__(self):
        pass

    def train(self):
        t = 0
        grad_avg = gradient_tool.grad_jit(self.target, self.data, self.theta_bar)
        # c
        #
        #
        #
        #