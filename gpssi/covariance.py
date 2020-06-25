import numpy as np
import scipy.linalg as linalg


class CovarianceRepresentation:

    def factorize_grid(self, shape: tuple):
        raise NotImplementedError()

    def sample(self, noise_vec):
        raise NotImplementedError()


class KroneckerCovariance(CovarianceRepresentation):

    def __init__(self, kernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.cov_kron_mats = None

    def factorize_grid(self, shape: tuple):
        self.cov_kron_mats = kronecker_grid_factorization(shape, self.kernel)

    def sample(self, noise_vec):
        return kronecker_matrix_vector_product(self.cov_kron_mats, noise_vec)


class FullCovariance(CovarianceRepresentation):

    def __init__(self, kernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.cov = None

    def factorize_grid(self, shape: tuple):
        self.cov = full_grid_factorization(shape, self.kernel)

    def sample(self, noise_vec):
        return self.cov @ noise_vec


def kronecker_matrix_vector_product(kron_matrices, x):
    x_res = x.copy()
    for kron_mat in kron_matrices[::-1]:
        n = kron_mat.shape[0]
        x_m = x_res.reshape(x_res.size // n, n).T
        ## instead of (as written in the paper)
        # x_m = x_res.reshape(n, x_res.size//n, order='F')
        z = kron_mat @ x_m
        z_vec = z.ravel()
        ## instead of (as written in the paper)
        # z = z.T  # probably not needed
        # z_vec = z.flatten(order='F')  # check if column-wise
        x_res = z_vec
    return x_res


def kronecker_grid_factorization(shape: tuple, kernel):
    kron_matrices = []
    for d in range(len(shape)):
        pos_d = np.arange(shape[d])[:, np.newaxis]
        cov_d = kernel(pos_d, pos_d)
        u_d = linalg.cholesky(cov_d)
        kron_matrices.append(u_d)
    return kron_matrices


def full_grid_factorization(shape: tuple, kernel: callable):
    pos = np.indices(shape).transpose((1, 2, 0))  # grid with voxel positions
    pos_vec = pos.reshape(-1, 2)
    cov = kernel(pos_vec, pos_vec)
    u = linalg.cholesky(cov)
    return u
