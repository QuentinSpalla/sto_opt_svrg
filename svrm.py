import numpy as np
import gradient_tool
from scipy.linalg import sqrtm


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
    def __init__(self, data, target, nbr_epoch, T, learning_rate, low_rank):
        self.data = data
        self.target = target
        self.nbr_epoch = nbr_epoch
        self.T = T
        self.learning_rate = learning_rate
        self.theta_k = np.zeros(data.shape[1])
        self.theta_bar = self.theta_k.copy()
        self.grad_history = np.ones(nbr_epoch)
        self.low_rank = low_rank

    def train(self, is_njit=True):
        t = self.T-1
        S = np.random.normal(size=(self.data.shape[1], self.low_rank))

        if is_njit:
            for i in range(self.nbr_epoch):
                if t == self.T - 1:
                    # calculate g(theta_bar)
                    self.theta_bar = self.theta_k.copy()
                    grad_avg = gradient_tool.grad_jit(self.target, self.data, self.theta_bar)
                    # calculate A, C
                    A = gradient_tool.hess_low_jit(self.data, S, self.target, self.theta_bar)
                    C = sqrtm(gradient_tool.get_curvature_matrix_jit(S, A))
                    # generate S, S_bar
                    S_bar = gradient_tool.mat_mul_jit(S, C)
                    A_bar = gradient_tool.mat_mul_jit(A, C)
                    # normalize hessian A_bar
                    self.grad_history[i] = np.linalg.norm(grad_avg)
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad_jit(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad_jit(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.get_first_term_cm_jit(A_bar, S_bar, self.theta_k, self.theta_bar, self.data,
                                                            self.target, i) \
                      + gradient_tool.get_second_term_cm_jit(A_bar, self.theta_k, self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                t += 1
        else:
            for i in range(self.nbr_epoch):
                if t == self.T - 1:
                    # calculate g(theta_bar)
                    self.theta_bar = self.theta_k.copy()
                    grad_avg = gradient_tool.grad(self.target, self.data, self.theta_bar)
                    # calculate A, C
                    A = gradient_tool.hess_low(self.data, S, self.target, self.theta_bar)
                    C = sqrtm(gradient_tool.get_curvature_matrix(S, A))
                    # generate S, S_bar
                    S_bar = gradient_tool.mat_mul(S, C)
                    A_bar = gradient_tool.mat_mul(A, C)
                    # normalize hessian A_bar
                    self.grad_history[i] = np.linalg.norm(grad_avg)
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.get_first_term_cm(A_bar, S_bar, self.theta_k, self.theta_bar, self.data,
                                                        self.target, i) \
                      + gradient_tool.get_second_term_cm(A_bar, self.theta_k, self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                t += 1
