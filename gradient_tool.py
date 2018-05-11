import numpy as np


# GRADIENT and HESSIAN
def sigmoid(x):
    return 1. / (1+np.exp(-x))


def grad(b, data, theta):
    """
    Computes gradient of function loss
    :param b: target
    :param data: features
    :param theta: weights
    :return: computed gradient
    """
    h = sigmoid(b * np.dot(data, theta))
    return np.dot(data.T, (h - 1) * b) / data.shape[0]


def partial_grad(b, data, theta, i):
    """
    Computes gradient of function loss for one sample
    :param b: target
    :param data: features
    :param theta: weights
    :param i: index of the sample
    :return: computed gradient
    """
    h = sigmoid(b[i] * np.dot(data[i], theta))
    return data[i] * (h-1)*b[i]


def get_loss(b, data, theta):
    """
    Computes the function loss
    :param b: target
    :param data: features
    :param theta: weights
    :return: computed loss
    """
    h = sigmoid(b*np.dot(data, theta))
    return -np.sum(np.log(h))


def partial_hess(data, b, theta, i):
    """
    Computes hessian of function loss for one sample
    :param data: features
    :param b: target
    :param theta: weights
    :param i: sample index
    :return: computed hessian
    """
    h = sigmoid(b[i] * np.dot(data[i], theta))
    return b[i]**2*h*(1-h)*np.dot(data[[i]].T, data[[i]])


def hess(data, b, theta):
    """
    Computes hessian of function loss
    :param data: features
    :param b: target
    :param theta: weights
    :return: computed hessian
    """
    temp = np.zeros((data.shape[1], data.shape[1]))
    for idx in range(data.shape[0]):
        temp += partial_hess(data, b, theta, idx)
    return temp / data.shape[0]


# CURVATURE MATCHING
def partial_hess_low(data, S, b, theta, i):
    """
    Computes low rank hessian of function loss for one sample
    :param data: features
    :param S: low rank matrix
    :param b: target
    :param theta: weights
    :param i: sample index
    :return: computed low rank hessian
    """
    h = sigmoid(b[i] * np.dot(data[i], theta))
    return b[i]**2*h*(1-h)*np.dot(data[[i]].T, np.dot(data[[i]], S))


def hess_low(data, S, b, theta):
    """
    Computes low rank hessian of function loss
    :param data: features
    :param S: low rank matrix
    :param b: target
    :param theta: weights
    :return: computed low rank hessian
    """
    temp = np.zeros((S.shape))
    for idx in range(data.shape[0]):
        temp += partial_hess_low(data, S, b, theta, idx)
    return temp / data.shape[0]


def get_second_term_cm(A_bar, theta, theta_bar):
    """
    Computes second term of the hessian approximation in the Curvature Matching method
    :param A_bar: low rank matrix
    :param theta: weights
    :param theta_bar: general weights
    """
    return np.dot(A_bar,
                  np.dot(A_bar.T,
                         theta-theta_bar))


def get_first_term_cm(A_bar, S_bar, theta, theta_bar, data, b, idx):
    """
    Computes first term of the hessian approximation in the Curvature Matching method
    :param A_bar: low rank matrix
    :param S_bar: low rank matrix
    :param theta: weights
    :param theta_bar: general weights
    :param data: features
    :param b: targets
    :param idx: sample index
    """
    return np.dot(A_bar,
                  np.dot(S_bar.T,
                         np.dot(partial_hess_low(data, S_bar, b, theta_bar, idx),
                                np.dot(A_bar.T,
                                       theta-theta_bar))))


def get_curvature_matrix(S, A):
    """
    :param S: low rank matrix
    :param A: low rank matrix
    """
    return np.linalg.pinv(np.dot(S.T, A))


# ACTION MATCHING
def get_first_term_am(A_bar, S_bar, theta, theta_bar, data, b, idx):
    """
    Computes second term of the hessian approximation in the Action Matching method
    :param A_bar: low rank matrix
    :param S_bar: low rank matrix
    :param theta: weights
    :param theta_bar: general weights
    :param data: features
    :param b: targets
    :param idx:  sample index
    """
    return np.dot(np.dot(A_bar,
                         np.dot(S_bar.T,
                                np.dot(partial_hess(data, b, theta_bar, idx),
                                       (np.identity(theta.shape[0]) - np.dot(S_bar, A_bar.T)))))
                  + np.dot(partial_hess_low(data, S_bar, b, theta_bar, idx),
                           A_bar.T),
                  theta - theta_bar)


# TOOLS
def mat_mul(mat1, mat2):
    return np.dot(mat1, mat2)
