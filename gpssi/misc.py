import numpy as np


def kron_mats_to_full(kron_mats):
    full = 1
    for kron_mat in kron_mats:
        full = np.kron(full, kron_mat)
    return full
