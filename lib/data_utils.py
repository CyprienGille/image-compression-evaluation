from typing import Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch as T
from torch.utils.data import Dataset


class KodakFolder(Dataset):
    """
    Image shape is either (768, 512, 3) or (512, 768, 3) --> 6x4 or 4x6 128x128 patches
    """

    def __init__(self, root: str, files_list=None):
        if files_list is None:
            self.files = sorted(Path(root).iterdir())
        else:
            self.files = sorted([Path(root) / k for k in files_list])

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, Tuple[int, int]]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path)).astype("float64")
        dims = img.shape
        if dims[0] == 768:
            pat_x, pat_y = 6, 4
        elif dims[0] == 512:
            pat_x, pat_y = 4, 6
        else:
            raise ValueError(
                f"KodakFolder dataset class expects (768, 512, 3) or (512, 768, 3) images but got {dims}."
            )

        img /= 255
        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, pat_x, 128, pat_y, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, (pat_x, pat_y)

    def __len__(self):
        return len(self.files)
