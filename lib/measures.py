import math

import cv2
import torch
from skimage import io
from skimage.metrics import structural_similarity as ssim


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


def SSIM(img_ref_path, img_comp_path):
    """Returns the structural similarity score between two images"""
    return ssim(*get_valid_input(img_ref_path, img_comp_path), channel_axis=-1)


def PSNR(img_ref_path, img_comp_path) -> float:
    return cv2.PSNR(*get_valid_input(img_ref_path, img_comp_path))

