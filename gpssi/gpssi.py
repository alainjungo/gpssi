import numpy as np
import GeodisTK as geo

import gpssi.covariance as c
import gpssi.kernel as k


def get_geodesic_map(np_img, np_seg, lmbda, iter=None, spacing=None):

    if not (np_img.ndim == 2 or np_img.ndim == 3):
        raise ValueError('The image must be 2- or 3-dimensional')
    if np_img.ndim != np_seg.ndim:
        raise ValueError('The image and segmentation must be of same dimensionality')

    mask = np_seg.astype(np.bool)

    if np_img.ndim == 3:
        if iter is None:
            iter = 4
        if spacing is None:
            raise ValueError('required to define the spacing')
        np_geo = geo.geodesic3d_raster_scan(np_img, mask, spacing, lmbda, iter)
        np_geo[mask] = -geo.geodesic3d_raster_scan(np_img, ~mask, spacing, lmbda, iter)[mask]
    else:
        if iter is None:
            iter = 2
        np_geo = geo.geodesic2d_raster_scan(np_img, mask, lmbda, iter)
        np_geo[mask] = -geo.geodesic2d_raster_scan(np_img, ~mask, lmbda, iter)[mask]
    return np_geo


def get_covariance(img_shape: tuple, kernel: k.Kernel, cov_repr: str = 'kron') -> c.CovarianceRepresentation:
    if cov_repr == 'kron':
        cov = c.KroneckerCovariance()
    elif cov_repr == 'full':
        cov = c.FullCovariance()
    else:
        raise ValueError(f'unknown covariance representation "{cov_repr}"')

    cov.factorize_grid(img_shape, kernel)
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


