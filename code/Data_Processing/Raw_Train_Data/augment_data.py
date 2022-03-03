import time

import torch
from PIL import Image
import torchvision.io as tio
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from tqdm import tqdm

from raw_dataset import RawImagesDataset

MULTIPLIER = 20 # make 20 images out of every one example


if __name__ == "__main__":
    # seed with time
    torch.manual_seed(int(time.time() * 1000))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # apply some affine transform + perspective shift + change brightness
    transforms = torch.nn.Sequential(
        trans.RandomAffine(degrees=(0, 45), translate=(0., 0.1), scale=(0.9, 1.1)),
        trans.RandomPerspective(distortion_scale=0.6, p=0.3),
        trans.ColorJitter(brightness=0.5)
    )
    # apparently, this makes it faster
    scripted_transforms = torch.jit.script(transforms).to(device)

    dataset = RawImagesDataset("D:\\facultate stuff\\licenta\\data\\train_imgs\\")
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for e in range(MULTIPLIER):
        dataset_augmented = []
        for i, (inputs, labels) in tqdm(enumerate(dataset_loader)):
            current_batch = scripted_transforms(inputs)
            dataset_augmented += current_batch

        for iindex, img in tqdm(enumerate(dataset_augmented)):
            (trans.ToPILImage()(img)).save("D:\\facultate stuff\\licenta\\data\\train_imgs_aug\\" + str(iindex + e * len(dataset)) + ".jpeg")
