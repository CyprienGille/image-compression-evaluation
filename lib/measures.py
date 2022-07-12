import numpy as np
import torch
from skimage import io
from skimage.metrics import structural_similarity as ssim
import cv2


def tensor_entropy(tensor):
    """ Determine the entropy of a tensor."""
    _, count = torch.unique(tensor.clone().detach(), return_counts=True)

    p = count / count.sum().item()

    return -(p * torch.log2(p)).sum().item()


def get_valid_input(img_ref_path, img_comp_path):
    img_ref = io.imread(img_ref_path)
    img_comp = io.imread(img_comp_path)

    s_ref = img_ref.shape
    s_comp = img_comp.shape
    if s_ref != s_comp:
        raise ValueError(
            f"Images {img_ref_path}({s_ref}) and {img_comp_path}({s_comp}) have different shapes."
        )

    yuv1 = cv2.cvtColor(img_ref, cv2.COLOR_RGB2YCrCb)
    yuv1[:, :, 1], yuv1[:, :, 2] = yuv1[:, :, 2], yuv1[:, :, 1]

    yuv2 = cv2.cvtColor(img_comp, cv2.COLOR_RGB2YCrCb)
    yuv2[:, :, 1], yuv2[:, :, 2] = yuv2[:, :, 2], yuv2[:, :, 1]

    return yuv1, yuv2


def compute_PSNR(yuv1, yuv2):
    """Computes the PSNR between two images in the YCrCb space"""

    PSNR = 0
    for k in range(3):  # over all channels
        diff = yuv2[:, :, k].astype(int) - yuv1[:, :, k].astype(int)
        mean_squared_error = 0
        for i in range(len(diff)):
            for j in range(len(diff[0])):
                mean_squared_error += diff[i, j] ** 2
        mean_squared_error /= len(diff)
        mean_squared_error /= len(diff[0])
        if k == 0:
            PSNR += 6 * np.log10((255 ** 2) / mean_squared_error)
        else:
            PSNR += 2 * np.log10((255 ** 2) / mean_squared_error)
    return PSNR


def PSNR(img_ref_path, img_comp_path):
    """Returns the PSNR between two images"""
    return compute_PSNR(*get_valid_input(img_ref_path, img_comp_path))


def SSIM(img_ref_path, img_comp_path):
    """Returns the structural similarity score between two images"""
    return ssim(*get_valid_input(img_ref_path, img_comp_path), channel_axis=-1)
