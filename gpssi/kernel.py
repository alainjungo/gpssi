import numpy as np


class Kernel:

    def __call__(self, x1, x2):
        raise NotImplementedError()


class RbfKernel(Kernel):

    def __init__(self, w0, w1, eps=None) -> None:
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.eps = eps

    def __call__(self, x1, x2):
        return rbf(x1, x2, self.w0, self.w1, self.eps)


def rbf(x1, x2, w0, w1, diag_eps=None):
    sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    res = w0 * np.exp(-sqdist / w1 ** 2)
    if diag_eps:
        res += np.eye(sqdist.shape[0])*diag_eps
    return res
