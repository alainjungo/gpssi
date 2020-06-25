import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gpssi


def main():
    np_img, np_seg = load_data('data/example_img.png', 'data/example_seg.png')
    np_img, np_seg = np_img[110:180, 100:180], np_seg[110:180, 100:180]

    np_geo = gpssi.get_geodesic_map(np_img, np_seg, lmbda=0.9, iter=2)

    # w1 = np.sqrt((np_seg > 0).sum() / np.pi)  # r of circle with equal area than the segmentation
    w1 = 4
    w0 = 1.5   # or (D/2)**2, where D s the expected distance in np_geo for the 95% CI
    kernel = gpssi.RbfKernel(w0, w1)

    cov = gpssi.get_covariance(np_img.shape, kernel, cov_repr='kron')
    # cov = gpssi.get_covariance(np_img.shape, kernel, cov_repr='full')

    samples = []
    for i in range(5):
        noise_vec = np.random.randn(np_geo.size)

        sample = gpssi.get_sample(np_geo, cov, noise_vec)
        samples.append(sample)

    # plotting
    plot_img(np_geo, 'jet', colorbar=True)
    plot_mask_overlay(np_img, np_seg)
    for sample in samples:
        plot_mask_overlay(np_img, sample)


def plot_img(arr, cmap='gray', colorbar=False):
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        fig.colorbar(im, ax=ax)
    plt.show()
    plt.close(fig)


def plot_mask_overlay(img, mask):
    fig, ax = plt.subplots()
    overlay(ax, img, mask)
    plt.show()
    plt.close(fig)


def overlay(ax, img, mask):
    ax.imshow(img, cmap='gray')
    ma_sample = np.ma.masked_equal(mask, 0)
    ax.imshow(ma_sample, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    ax.set_axis_off()


def load_data(img_path, seg_path):
    np_img = np.asarray(Image.open(img_path)).astype(np.float32)
    np_seg = np.asarray(Image.open(seg_path)).astype(np.uint8) // 255
    return np_img, np_seg


if __name__ == '__main__':
    main()
