import os

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.io as tio
import torchvision.transforms as trans
from PIL import Image

def load_dataset_raw():
    dataset_path = "D:\\facultate stuff\\licenta\\data\\train_imgs\\"
    dataset_labels_path = "D:\\facultate stuff\\licenta\\data\\train_labels_RAW.txt"

    dataset = []
    dataset_labels = []
    with open(dataset_labels_path, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            fp, cls, *coords = line.split(",")
            dataset.append(tio.read_image(dataset_path + str(index) + ".jpg"))
            dataset_labels.append(cls)

    return dataset, dataset_labels


class RawImagesDataset(Dataset):
    def __init__(self, path, transform=None):
        dataset, dataset_labels = load_dataset_raw()
        self.img_labels = dataset_labels
        self.img_dir = path
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, str(item) + ".jpg")
        img = trans.PILToTensor()(Image.open(img_path).resize((64, 32))).to('cuda')
        label = self.img_labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label