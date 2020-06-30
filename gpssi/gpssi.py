import numpy as np
import GeodisTK as geo

import gpssi.covariance as c
import gpssi.kernel as k


def get_geodesic_map(np_img, np_seg, lmbda, iter=2):
    mask = np_seg.astype(np.bool)

    np_geo = geo.geodesic2d_raster_scan(np_img, mask, lmbda, iter)
    np_geo[mask] = -geo.geodesic2d_raster_scan(np_img, ~mask, lmbda, iter)[mask]
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


