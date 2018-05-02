import numpy as np
from numba import njit


@njit(parallel=True)
def grad_jit(b, A, theta):
    return - np.dot(A.T, b - np.dot(A, theta)) / A.shape[0]


@njit(parallel=True)
def partial_grad_jit(b, A, theta, i):
    return - A[i] * (b[i] - np.dot(A[i], theta))


@njit(parallel=True)
def get_loss(b, A, theta):
    return 0.5 * np.sum((b - np.dot(A, theta)) ** 2)

