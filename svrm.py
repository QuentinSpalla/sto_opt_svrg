import numpy as np
import gradient_tool
from scipy.linalg import sqrtm
import datetime as dt


class SVRG():
    def __init__(self, data, target, nbr_epoch, T, learning_rate, nbr_seconds):
        self.data = data
        self.target = target
        self.nbr_epoch = nbr_epoch
        self.T = T
        self.learning_rate = learning_rate
        self.theta_k = np.zeros(data.shape[1])
        self.theta_bar = self.theta_k.copy()
        self.nbr_epoch = nbr_epoch
        self.grad_history = []
        self.loss_history = []
        self.nbr_seconds = nbr_seconds

    def train(self, is_with_datapass=True):
        t = self.T-1

        if is_with_datapass:
            #self.grad_history = np.ones(self.nbr_epoch)
            #self.loss_history = np.ones(self.nbr_epoch)
            svrg_update = 1
            for i in range(self.nbr_epoch):
                # .. save gradient (for plotting purposes) every epoch ..
                if t == self.T-1:
                    self.theta_bar = self.theta_k.copy()
                    grad_avg = gradient_tool.grad(self.target, self.data, self.theta_bar)
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                # .. pick random sample ..
                idx = np.random.randint(0, self.data.shape[0])

                # .. compute and apply SVRG update rule ..
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                self.theta_k = self.theta_k - self.learning_rate * svrg_update
                #self.loss_history.append(np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                t += 1
        else:
            i = 0
            end_time = dt.datetime.now() + dt.timedelta(seconds=self.nbr_seconds)
            while dt.datetime.now() < end_time:
                # .. save gradient (for plotting purposes) every epoch ..
                if t == self.T - 1:
                    self.theta_bar = self.theta_k.copy()
                    grad_avg = gradient_tool.grad(self.target, self.data, self.theta_bar)
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                # .. pick random sample ..
                idx = np.random.randint(0, self.data.shape[0])

                # .. compute and apply SVRG update rule ..
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                self.theta_k = self.theta_k - self.learning_rate * svrg_update
                t += 1
                i += 1


class CM():
    def __init__(self, data, target, nbr_epoch, T, learning_rate, low_rank, nbr_seconds):
        self.data = data
        self.target = target
        self.nbr_epoch = nbr_epoch
        self.T = T
        self.learning_rate = learning_rate
        self.theta_k = np.random.uniform(low=-1. / np.sqrt(data.shape[1]), high=1. / np.sqrt(data.shape[1]), size=data.shape[1])
        self.theta_bar = self.theta_k.copy()
        self.grad_history = []
        self.loss_history = []
        self.low_rank = low_rank
        self.nbr_seconds = nbr_seconds
        self.nbr_epoch = nbr_epoch

    def train(self, is_with_datapass=True):
        t = self.T-1
        S = np.random.normal(size=(self.data.shape[1], self.low_rank))

        if is_with_datapass:
            #self.grad_history = np.ones(self.nbr_epoch)
            #self.loss_history = np.ones(self.nbr_epoch)
            d_t = 1
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
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(
                        np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.get_first_term_cm(A_bar, S_bar, self.theta_k, self.theta_bar, self.data,
                                                        self.target, idx) \
                      + gradient_tool.get_second_term_cm(A_bar, self.theta_k, self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                #self.loss_history[i] = np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k))
                t += 1
        else:
            i = 0
            end_time = dt.datetime.now() + dt.timedelta(seconds=self.nbr_seconds)
            while dt.datetime.now() < end_time:
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
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(
                        np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.get_first_term_cm(A_bar, S_bar, self.theta_k, self.theta_bar, self.data,
                                                        self.target, idx) \
                      + gradient_tool.get_second_term_cm(A_bar, self.theta_k, self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                if t == 0:
                    self.grad_history.append(np.linalg.norm(d_t))
                #self.loss_history.append(np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                t += 1
                i += 1


class SVRG2():
    def __init__(self, data, target, nbr_epoch, T, learning_rate, nbr_seconds):
        self.data = data
        self.target = target
        self.nbr_epoch = nbr_epoch
        self.T = T
        self.learning_rate = learning_rate
        self.theta_k = np.random.uniform(low=-1. / np.sqrt(data.shape[1]), high=1. / np.sqrt(data.shape[1]), size=data.shape[1])
        self.theta_bar = self.theta_k.copy()
        self.grad_history = []
        self.loss_history = []
        self.nbr_seconds = nbr_seconds
        self.nbr_epoch = nbr_epoch

    def train(self, is_with_datapass=True):
        t = self.T-1

        if is_with_datapass:
            #self.grad_history = np.ones(self.nbr_epoch)
            #self.loss_history = np.ones(self.nbr_epoch)
            for i in range(self.nbr_epoch):
                if t == self.T - 1:
                    # calculate g(theta_bar)
                    self.theta_bar = self.theta_k.copy()
                    grad_avg = gradient_tool.grad(self.target, self.data, self.theta_bar)
                    hess_avg = gradient_tool.hess(self.data, self.target, self.theta_bar)
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(
                        np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.partial_hess(self.data, self.target, self.theta_k, idx).dot(self.theta_k-self.theta_bar) \
                      + hess_avg.dot(self.theta_k-self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                #self.loss_history[i] = np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k))
                t += 1
        else:
            i = 0
            end_time = dt.datetime.now() + dt.timedelta(seconds=self.nbr_seconds)
            while dt.datetime.now() < end_time:
                if t == self.T - 1:
                    # calculate g(theta_bar)
                    self.theta_bar = self.theta_k.copy()
                    grad_avg = gradient_tool.grad(self.target, self.data, self.theta_bar)
                    hess_avg = gradient_tool.hess(self.data, self.target, self.theta_bar)
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(
                        np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.partial_hess(self.data, self.target, self.theta_k, idx).dot(
                    self.theta_k - self.theta_bar) \
                      + hess_avg.dot(self.theta_k - self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                #self.loss_history.append(np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                t += 1
                i += 1


class AM():
    def __init__(self, data, target, nbr_epoch, T, learning_rate, low_rank, nbr_seconds):
        self.data = data
        self.target = target
        self.nbr_epoch = nbr_epoch
        self.T = T
        self.learning_rate = learning_rate
        self.theta_k = np.random.uniform(low=-1. / np.sqrt(data.shape[1]), high=1. / np.sqrt(data.shape[1]), size=data.shape[1])
        self.theta_bar = self.theta_k.copy()
        self.grad_history = []
        self.loss_history = []
        self.low_rank = low_rank
        self.nbr_seconds = nbr_seconds
        self.nbr_epoch = nbr_epoch

    def train(self, is_with_datapass=True):
        t = self.T-1
        S = np.random.normal(size=(self.data.shape[1], self.low_rank))

        if is_with_datapass:
            #self.grad_history = np.ones(self.nbr_epoch)
            #self.loss_history = np.ones(self.nbr_epoch)
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
                    #self.grad_history[i] = np.linalg.norm(grad_avg)
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(
                        np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.get_first_term_am(A_bar, S_bar, self.theta_k, self.theta_bar, self.data,
                                                        self.target, idx) \
                      + gradient_tool.get_second_term_cm(A_bar, self.theta_k, self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                #self.loss_history[i] = np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k))
                t += 1
        else:
            i = 0
            end_time = dt.datetime.now() + dt.timedelta(seconds=self.nbr_seconds)
            while dt.datetime.now() < end_time:
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
                    self.grad_history.append(np.linalg.norm(grad_avg))
                    self.loss_history.append(
                        np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                    t = 0

                idx = np.random.randint(0, self.data.shape[0])
                cur_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_k, idx)
                prev_grad = gradient_tool.partial_grad(self.target, self.data, self.theta_bar, idx)
                svrg_update = cur_grad - prev_grad + grad_avg
                d_t = svrg_update \
                      - gradient_tool.get_first_term_am(A_bar, S_bar, self.theta_k, self.theta_bar, self.data,
                                                        self.target, idx) \
                      + gradient_tool.get_second_term_cm(A_bar, self.theta_k, self.theta_bar)
                self.theta_k = self.theta_k - self.learning_rate * d_t
                #self.loss_history.append(np.linalg.norm(gradient_tool.get_loss(self.target, self.data, self.theta_k)))
                t += 1
                i += 1