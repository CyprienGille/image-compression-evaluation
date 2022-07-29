import pathlib
from lib.data_utils import KodakFolder
from lib.measures import PSNR, SSIM, tensor_entropy
from lib.tensor_utils import quantize_tensor, dequantize_tensor
from models.cae_32x32x32_zero_pad_comp import CAE
from models.cae_lightning import LightningCAE

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OpenMP duplicate issues


def get_data_point(
    model,
    dataloader,
    nbits=-1,
    save_decoded_dir="./temp/",
    reference_dir="./datasets/kodak/",
):
    """Get the bitrate, psnr and ssim for a given number of quantization bits,
    as averaged over all images in the given dataloader
    Note: nbits=-1 -> No quantization"""

    os.makedirs(save_decoded_dir, exist_ok=True)

    all_bitrates = []
    all_psnrs = []
    all_ssims = []

    for idx, data in enumerate(dataloader):
        _, patches, (nb_pat_x, nb_pat_y) = data
        # put data on GPU if available
        patches = patches.to(DEVICE, non_blocking=True)

        # complete output tensor
        out = torch.zeros(nb_pat_x, nb_pat_y, 3, 128, 128)
        # tensor to store entropies per patch
        patch_ent = np.zeros((nb_pat_x, nb_pat_y))

        # iterate over patches
        for i in range(nb_pat_x):
            for j in range(nb_pat_y):
                x = patches[0, :, i, j, :, :].unsqueeze(0)
                encoded, decoded = model(x)

                if nbits > 0:  # if compression
                    # quantize the encoded data, and pass it to the decoder
                    encoded, scale, zero_pt = quantize_tensor(
                        encoded.cpu(), num_bits=nbits
                    )
                    encoded = encoded.to(DEVICE)
                    decoded = model.decode(dequantize_tensor(encoded, scale, zero_pt))

                out[i, j] = decoded.data
                # entropy of a patch
                patch_ent[i, j] = tensor_entropy(encoded)

        all_bitrates.append(np.average(patch_ent) * 2 / 3)  # bpp = 2/3 * entropy

        # reshape output into an image
        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (nb_pat_x * 128, nb_pat_y * 128, 3))
        out = np.transpose(out, (2, 0, 1))
        out.unsqueeze(0)
        out.clamp(0, 1)

        save_name = f"{save_decoded_dir}Decoded_{idx}_{nbits}bits.png"
        save_image(out, save_name)
        # this f-string allows to get double digits for all images
        # ex: kodim01.png
        # without getting for example kodim015.png
        ref_name = f"kodim{'0'*(idx<9)}{idx+1}.png"
        all_psnrs.append(PSNR(reference_dir + "/" + ref_name, save_name))
        all_ssims.append(SSIM(reference_dir + "/" + ref_name, save_name))

    # return average over all images
    return np.average(all_bitrates), np.average(all_psnrs), np.average(all_ssims)


if __name__ == "__main__":
    if os.name != "posix":  # if on windows
        pathlib.PosixPath = pathlib.WindowsPath  # to use models trained on linux

    PLOTS_DIR = "./plots/"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # all models to plot
    all_models = [
        "trainComp_200_0.0_Lightning_initial",
        "trainComp_100_0.0_Flickr_initial",
        "trainComp_200_0.0_halfprojNB_Flickr_L11_500.0",
    ]
    # all n_bits for quantization to plot
    quantization_bits = [2, 3, 4, 6, 8, 10, 16]

    model_dir = "./trained_models"
    img_dir = "./datasets/kodak"

    # Prepare data
    dataset = KodakFolder(root=img_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2 subplots, 1 for PSNR and 1 for MSSIM
    plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    for model_name in all_models:
        bitrate_points, psnr_points, mssim_points = [], [], []
        model_path = f"{model_dir}/{model_name}/checkpoint/best_model.pth"
        if model_name.find("Lightning") != -1:
            # Lightning-trained model, re-used here in pure pytorch
            model = LightningCAE()
            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE)["state_dict"]
            )
        else:
            model = CAE()
            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE)["model_state_dict"]
            )
        model.eval()
        model = model.to(DEVICE)

        for bits in tqdm(quantization_bits):
            bpp, psnr, mssim = get_data_point(
                model, dataloader, nbits=bits, reference_dir=img_dir
            )
            bitrate_points.append(bpp)
            psnr_points.append(psnr)
            mssim_points.append(mssim)

        plt.subplot(1, 2, 1)
        plt.plot(bitrate_points, psnr_points, label=model_name, marker="x")
        plt.subplot(1, 2, 2)
        plt.plot(bitrate_points, mssim_points, label=model_name, marker="x")

    plt.subplot(1, 2, 1)
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.subplot(1, 2, 2)
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("MSSIM")
    plt.legend()
    plt.savefig(
        "plots/PSNR_MSSIM_bpp.png", dpi=400, facecolor="white", bbox_inches="tight"
    )
    plt.show()

