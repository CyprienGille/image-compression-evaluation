#%%
import pathlib
from lib.data_utils import KodakFolder, ImagesDataset
from models.cae_32x32x32_zero_pad_comp import CAE
from models.cae_lightning import LightningCAE

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from PIL import Image
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OpenMP duplicate issues

#%%
def process_imgs_C(model, dataloader, save_decoded_dir="./temp/CC/"):
    """Just encode and decode the images, save them
    Original Method"""

    os.makedirs(save_decoded_dir, exist_ok=True)

    for idx, data in enumerate(tqdm(dataloader)):
        _, patches, (nb_pat_x, nb_pat_y) = data
        patches = patches.to(DEVICE, non_blocking=True)

        out = torch.zeros(nb_pat_x, nb_pat_y, 3, 128, 128)

        # iterate over patches
        for i in range(nb_pat_x):
            for j in range(nb_pat_y):
                x = patches[0, :, i, j, :, :].unsqueeze(0)
                _, decoded = model(x)

                out[i, j] = decoded.data

        # reshape output into an image
        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (nb_pat_x * 128, nb_pat_y * 128, 3))
        out = np.transpose(out, (2, 0, 1))
        out.unsqueeze(0)
        out.clamp(0, 1)

        save_name = f"{save_decoded_dir}Decoded_{idx}.png"
        save_image(out, save_name)


#%%
def process_imgs_G(model, dataloader, save_decoded_dir="./temp/GG/"):
    """Just encode and decode the images, save them
    M. Guyard Method"""

    os.makedirs(save_decoded_dir, exist_ok=True)

    for idx, data in enumerate(tqdm(dataloader)):

        x, _ = data
        x = x.to(DEVICE, non_blocking=True)

        out = torch.zeros(x.shape, device=DEVICE)

        # iterate over patches
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                _, decoded = model(x[:, :, i, j, :, :])
                out[:, :, i, j, :, :] = decoded

        # reshape output into an image
        im0 = out[0]
        im2 = torch.permute(im0, (0, 1, 3, 2, 4))
        im3 = torch.reshape(
            im2, (3, im2.shape[1] * im2.shape[2], im2.shape[3] * im2.shape[4])
        )
        # im4.unsqueeze(0)
        im4 = im3.clamp(0, 1)
        save_name = f"{save_decoded_dir}Decoded_{idx}.png"
        # Image.fromarray(np.uint8(im4.cpu().detach().numpy())).save(save_name)

        save_image(im4, save_name)


#%%
if os.name != "posix":
    pathlib.PosixPath = pathlib.WindowsPath  # to use models trained on linux
model_dir = "./trained_models"
img_dir = "./datasets/kodak"


#%%
## MODEL CYPRIEN METHOD CYPRIEN
model_name = "trainComp_100_0.0_Flickr_initial"
model_path = f"{model_dir}/{model_name}/checkpoint/best_model.pth"
model = CAE()
model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model_state_dict"])
model.eval()
model = model.to(DEVICE)

dataset = KodakFolder(root=img_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

process_imgs_C(model, dataloader, save_decoded_dir="./temp/CC/")


#%%
## MODEL GUYARD METHOD CYPRIEN
model_name = "trainComp_100_0.0_Lightning_initial"
model_path = f"{model_dir}/{model_name}/checkpoint/best_model.pth"

model = LightningCAE()
model.load_state_dict(torch.load(model_path, map_location=DEVICE)["state_dict"])
model.eval()
model = model.to(DEVICE)

dataset = KodakFolder(root=img_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

process_imgs_C(model, dataloader, save_decoded_dir="./temp/CG/")

#%%
## MODEL GUYARD METHOD GUYARD
model_name = "trainComp_100_0.0_Lightning_initial"
model_path = f"{model_dir}/{model_name}/checkpoint/best_model.pth"

dataset = ImagesDataset(root=img_dir, x_patch=4, y_patch=6)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = LightningCAE()
model.load_state_dict(torch.load(model_path, map_location=DEVICE)["state_dict"])
model.eval()
model = model.to(DEVICE)

process_imgs_G(model, dataloader, save_decoded_dir="./temp/GG/")

#%%
## MODEL CYPRIEN METHOD GUYARD
model_name = "trainComp_100_0.0_Flickr_initial"
model_path = f"{model_dir}/{model_name}/checkpoint/best_model.pth"
model = CAE()
model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model_state_dict"])
model.eval()
model = model.to(DEVICE)

dataset = ImagesDataset(root=img_dir, x_patch=4, y_patch=6)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

process_imgs_G(model, dataloader, save_decoded_dir="./temp/GC/")

#%%
