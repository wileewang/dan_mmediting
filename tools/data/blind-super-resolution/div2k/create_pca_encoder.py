"""
    Create PCA Encoders to reduce kernel size from [1, L, L] to [1, l], e.g. from [1, 21, 21] to [1, 10]

    References:
        @article{luo2020unfolding,
              title={Unfolding the Alternating Optimization for Blind Super Resolution},
              author={Luo, Zhengxiong and Huang, Yan and Li, Shang and Wang, Liang and Tan, Tieniu},
              journal={Advances in Neural Information Processing Systems (NeurIPS)},
              volume={33},
              year={2020}
        }
        @misc{luo2021endtoend,
              title={End-to-end Alternating Optimization for Blind Super Resolution},
              author={Zhengxiong Luo and Yan Huang and Shang Li and Liang Wang and Tieniu Tan},
              year={2021},
              eprint={2105.06878},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
"""

import torch
import numpy as np
import argparse


def main(args):
    # kernel
    number = args.kernel_number
    kernel_size = int(args.kernel_size)
    kernel_sigma_min = args.kernel_sigma_min
    kernel_sigma_max = args.kernel_sigma_max
    isotropic_rate = args.isotropic_rate
    random_disturb = args.random_disturb

    # reduced kernel
    coded_length = args.coded_length

    # metrics path
    save_path = args.save_path

    kernels = generate_kernels(number=number,
                               kernel_size=kernel_size,
                               sigma_min=kernel_sigma_min,
                               sigma_max=kernel_sigma_max,
                               isotropic_rate=isotropic_rate,
                               random_disturb=random_disturb
                               )

    b = np.size(kernels, 0)
    kernels = kernels.reshape((b, -1))
    pca_matrix = pca(kernels, coded_length).float()
    torch.save(pca_matrix, save_path)


def generate_kernels(number,
                     kernel_size,
                     sigma_min,
                     sigma_max,
                     isotropic_rate,
                     random_disturb):
    if isotropic_rate == 1:

        sigma = np.random.uniform(sigma_min, sigma_max, (number, 1, 1))
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xx = xx[None].repeat(number, 0)
        yy = yy[None].repeat(number, 0)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)

    else:

        sigma_x = np.random.uniform(sigma_min, sigma_max, (number, 1, 1))
        sigma_y = np.random.uniform(sigma_min, sigma_max, (number, 1, 1))

        D = np.zeros((number, 2, 2))
        D[:, 0, 0] = sigma_x.squeeze() ** 2
        D[:, 1, 1] = sigma_y.squeeze() ** 2

        radians = np.random.uniform(-np.pi, np.pi, (number))
        mask_iso = np.random.uniform(0, 1, (number)) < isotropic_rate
        radians[mask_iso] = 0
        sigma_y[mask_iso] = sigma_x[mask_iso]

        U = np.zeros((number, 2, 2))
        U[:, 0, 0] = np.cos(radians)
        U[:, 0, 1] = -np.sin(radians)
        U[:, 1, 0] = np.sin(radians)
        U[:, 1, 1] = np.cos(radians)
        sigma = np.matmul(U, np.matmul(D, U.transpose(0, 2, 1)))
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(
            kernel_size, kernel_size, 2)
        xy = xy[None].repeat(number, 0)
        inverse_sigma = np.linalg.inv(sigma)[:, None, None]
        kernel = np.exp(
            -0.5
            * np.matmul(
                np.matmul(xy[:, :, :, None], inverse_sigma), xy[:, :, :, :, None]
            )
        )
        kernel = kernel.reshape(number, kernel_size, kernel_size)
        if random_disturb:
            kernel = kernel + np.random.uniform(0, 0.25, (1, kernel_size, kernel_size)) * kernel
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)

    return kernel


def pca(data, length=2):
    x = torch.from_numpy(data)
    x_mean = torch.mean(x, 0)
    x = x - x_mean.expand_as(x)
    u, s, v = torch.svd(torch.t(x))
    return u[:, :length]



def parse_args():
    parser = argparse.ArgumentParser(
        description='Create PCA Encoder Matrix',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--kernel_size',
        nargs='?',
        default=31,
        help='size of newly generated kernels')

    parser.add_argument(
        '--isotropic_rate',
        nargs='?',
        default=0.0,
        help='whether using regular gaussian blur kernel or not'
             '1.0 means isotropic and 0.0 means anisotropic')

    parser.add_argument(
        '--random_disturb',
        nargs='?',
        default=True,
        help='whether add a disturbing term specified in DAN paper to kernels')

    parser.add_argument(
        '--kernel_number',
        nargs='?',
        default=30000,
        help='number of newly generated kernels')

    parser.add_argument(
        '--kernel_sigma_min',
        nargs='?',
        default=0.5,
        help='min value of sigma of kernel,'
             'works in independent mode')

    parser.add_argument(
        '--kernel_sigma_max',
        nargs='?',
        default=6.0,
        help='max value of sigma of kernel,'
             'works in independent mode')

    parser.add_argument(
        '--coded_length',
        nargs='?',
        default=10,
        help='size of reduced kernel')

    parser.add_argument(
        '--save_path',
        default='pca_matrix.pth',
        help='path to save PCA encoder matrix')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # create PCA encoder
    main(args)
