import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import gpssi


def main():
    np_img, np_seg, spacing = load_data('data/example_img.nii.gz', 'data/example_seg.nii.gz')
    np_img = np_img / np_img.max() * 255  # normalize to be consistent with 2d example
    np_img, np_seg = np_img[0:144, 42:220, 47:192], np_seg[0:144, 42:220, 47:192]

    np_geo = gpssi.get_geodesic_map(np_img, np_seg, lmbda=0.9, iter=4, spacing=spacing)
    np_geo[np_img == 0] = np_geo.max()  # set background distance to max

    # r of sphere with equal area than the segmentation
    w1 = ((3 * (np_seg > 0).sum()) / (4 * np.pi))**(1/3)  # radius r=(3/4*V/pi)^(1/2)
    w0 = 3   # or (D/2)**2, where D s the expected distance in np_geo for the 95% CI
    kernel = gpssi.RbfKernel(w0, w1, eps=1e-8)

    cov = gpssi.get_covariance(np_img.shape, kernel, cov_repr='kron')

    samples = []
    for i in range(5):
        noise_vec = np.random.randn(np_geo.size)

        sample = gpssi.get_sample(np_geo, cov, noise_vec)
        samples.append(sample)

    # plot slice with largest gt
    sums = np_seg.sum(axis=(1, 2))
    slice_idx = np.argmax(sums)
    plot_img(np_geo[slice_idx], 'jet', colorbar=True)
    plot_mask_overlay(np_img[slice_idx], np_seg[slice_idx])
    for sample in samples:
        plot_mask_overlay(np_img[slice_idx], sample[slice_idx])


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
    img = sitk.ReadImage(img_path)

    np_img = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = img.GetSpacing()[::-1]  # invert since x and z  dims are swapped from sitk to numpy

    np_seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.uint8)
    return np_img, np_seg, spacing


if __name__ == '__main__':
    main()
