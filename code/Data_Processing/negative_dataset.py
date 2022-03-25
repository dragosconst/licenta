import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NegativeExamples(Dataset):

    """
    FP is expected to be an absolute filepath of the folder which contains the images.
    """
    def __init__(self, fp):
        self.fp = fp
        self.img_fps = sorted(glob.glob(os.path.join(fp, "*.jpg")))

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx[0]
        img = Image.open(self.img_fps[idx]).convert("RGB")
        targets = {"boxes": [], "labels": [], "image_id": [], "area": [], "iscrowd": []}
        targets["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        targets["area"] = torch.as_tensor(0, dtype=torch.float32)
        targets["labels"] = torch.as_tensor([], dtype=torch.int64)
        targets["image_id"] = torch.Tensor([idx]).type(torch.int64)
        targets["iscrowd"] = torch.as_tensor([0])
        img = transforms.PILToTensor()(img)
        img = img[:3, :, :] # drop alpha channel, if it exists
        img = transforms.ConvertImageDtype(torch.float32)(img)
        return img, targets

    def __len__(self):
        return len(self.img_fps)

