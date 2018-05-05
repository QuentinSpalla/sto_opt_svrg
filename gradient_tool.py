import numpy as np
from numba import njit


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def grad(b, data, theta):
    # return - np.dot(A.T, b - np.dot(data, theta)) / A.shape[0]
    h = sigmoid(b * np.dot(data, theta))
    return np.dot(data.T, (h - 1) * b)


def partial_grad(b, data, theta, i):
    # return - A[i] * (b[i] - np.dot(A[i], theta)) #:Least Square Loss
    h = sigmoid(b[i] * np.dot(data[i, None], theta))
    return data[i] * (h-1)*b[i]


def get_loss(b, data, theta):
    # return 0.5 * np.sum((b - np.dot(data, theta)) ** 2) #: Least Square Loss
    h = sigmoid(b*np.dot(data, theta))
    return -np.sum(np.log(h)) #: Logistic Loss


def partial_hess_low(data, S, b, theta, i):
    # return np.dot(data[i].T, np.dot(data[i], S)) #: Least Square Loss
    h = sigmoid(b[i] * np.dot(data[i, None], theta))
    return b[i]**2*h*(1-h)*np.dot(data[i, None].T, np.dot(data[i, None], S))


def hess_low(data, S, b, theta):
    temp = np.zeros((S.shape))
    for idx in range(data.shape[0]):
        temp += partial_hess_low(data, S, b, theta, idx)
    return temp


def get_curvature_matrix(S, A):
    return np.linalg.pinv(np.dot(S.T, A))


def mat_mul(mat1, mat2):
    return np.dot(mat1, mat2)


def get_second_term_cm(A_bar, theta, theta_bar):
    return np.dot(A_bar,
                  np.dot(A_bar.T,
                         theta-theta_bar))


def get_first_term_cm(A_bar, S_bar, theta, theta_bar, data, b, idx):
    return np.dot(A_bar,
                  np.dot(S_bar.T,
                         np.dot(partial_hess_low(data, S_bar, b, theta_bar, idx),
                                np.dot(A_bar.T,
                                       theta-theta_bar))))


# PARALLEL FUNCTIONS
@njit(parallel=True)
def sigmoid_jit(x):
    return 1 / (1+np.exp(-x))


@njit(parallel=True)
def grad_jit(b, data, theta):
    # return - np.dot(A.T, b - np.dot(data, theta)) / A.shape[0]
    h = sigmoid_jit(b * np.dot(data, theta))
    return np.dot(data.T, (h - 1) * b)


@njit(parallel=True)
def partial_grad_jit(b, data, theta, i):
    # return - A[i] * (b[i] - np.dot(A[i], theta)) #:Least Square Loss
    h = sigmoid_jit(b[i] * np.dot(data[i], theta))
    return data[i] * (h-1)*b[i]


@njit(parallel=True)
def get_loss_jit(b, data, theta):
    # return 0.5 * np.sum((b - np.dot(data, theta)) ** 2) #: Least Square Loss
    h = sigmoid_jit(b*np.dot(data, theta))
    return -np.sum(np.log(h)) #: Logistic Loss


@njit(parallel=True)
def partial_hess_low_jit(data, S, b, theta, i):
    # return np.dot(data[i].T, np.dot(data[i], S)) #: Least Square Loss
    h = sigmoid_jit(b[i] * np.dot(data[i], theta))
    return b[i]**2*h*(1-h)*np.dot(data[i].reshape(S.shape[0], 1), np.dot(data[i].reshape(1, S.shape[0]), S))


@njit(parallel=True)
def hess_low_jit(data, S, b, theta):
    temp = np.zeros((S.shape))
    for idx in range(data.shape[0]):
        temp += partial_hess_low_jit(data, S, b, theta, idx)
    return temp


@njit(parallel=True)
def get_curvature_matrix_jit(S, A):
    return np.linalg.pinv(np.dot(S.T, A))


@njit(parallel=True)
def mat_mul_jit(mat1, mat2):
    return np.dot(mat1, mat2)


@njit(parallel=True)
def get_second_term_cm_jit(A_bar, theta, theta_bar):
    return np.dot(A_bar,
                  np.dot(A_bar.T,
                         theta-theta_bar))


@njit(parallel=True)
def get_first_term_cm_jit(A_bar, S_bar, theta, theta_bar, data, b, idx):
    return np.dot(A_bar,
                  np.dot(S_bar.T,
                         np.dot(partial_hess_low_jit(data, S_bar, b, theta_bar, idx),
                                np.dot(A_bar.T,
                                       theta-theta_bar))))
