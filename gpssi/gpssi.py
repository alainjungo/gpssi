import numpy as np
import scipy.ndimage.morphology as morphology
import GeodisTK as geo

import gpssi.covariance as c
import gpssi.kernel as k


def get_geodesic_map(np_img, np_seg, lmbda, iter=2, return_boarder=False):
    mask = np_seg.astype(np.bool)

    conn = morphology.generate_binary_structure(mask.ndim, 1)
    np_boarder = mask ^ morphology.binary_erosion(mask, conn)

    np_geo = geo.geodesic2d_raster_scan(np_img, np_boarder, lmbda, iter)
    np_geo[mask] = -np_geo[mask]

    if return_boarder:
        return np_geo, np_boarder
    return np_geo


def get_covariance(img_shape: tuple, kernel: k.Kernel, cov_repr: str = 'kron') -> c.CovarianceRepresentation:
    if cov_repr == 'kron':
        cov = c.KroneckerCovariance(kernel)
    elif cov_repr == 'full':
        cov = c.FullCovariance(kernel)
    else:
        raise ValueError(f'unknown covariance representation "{cov_repr}"')

    cov.factorize_grid(img_shape)
    return cov


def get_sample(geo_map: np.ndarray, cov: c.CovarianceRepresentation,
               noise_vec: np.ndarray = None, return_geo_sample=False):
    if noise_vec is None:
        noise_vec = np.random.randn(geo_map.size)

    var = cov.sample(noise_vec)

    geo_sample = geo_map.ravel() + var
    geo_sample = geo_sample.reshape(geo_map.shape)

    seg_sample = geo_sample <= 0

    if return_geo_sample:
        return seg_sample, geo_sample
    return seg_sample


