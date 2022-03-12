from typing import List, Optional, Dict, Tuple
import glob
import os

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
from tqdm import tqdm

from Utils.file_utils import get_image_files, get_random_bg_img
from Utils.trans import RandomAffineBoxSensitive, RandomPerspectiveBoxSensitive, MyCompose
from Data_Processing.Raw_Train_Data.raw import parse_xml, possible_classes
"""
Generate a dataset of a certain size from a given dataset of cropped cards and another dataset of random backgrounds.

"""

IM_HEIGHT = 1080
IM_WIDTH = 1900
def generate_random_image(*cards, bg_image: Image.Image) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    """

    :param cards: tuples of (img, data), where img is the pic of the card and data represents the relevant labeling data
    :param bg_image: a random background image
    :return:
    """
    assert cards is not None

    card_masks = []
    transforms = MyCompose(
       (RandomAffineBoxSensitive(degrees=(0, 45), scale=(0.5, 1.5), prob=0.5),
        RandomPerspectiveBoxSensitive(dist_scale=0.3, prob=0.2))
    )
    # create the card masks, they will be a couple of black images with cards on them after various transforms
    for (img, data) in tqdm(cards):
        img_full = np.zeros((IM_HEIGHT, IM_WIDTH, 4), dtype=np.uint8)
        imw, imh = img.size
        x = torch.randint(low=20, high=IM_WIDTH - imw//2, size=(1,)).item()
        y = torch.randint(low=20, high=IM_HEIGHT - imh//2, size=(1,)).item()
        boxes = data["boxes"]
        new_boxes = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 + x >= IM_WIDTH or y1 + y >= IM_HEIGHT or x2 + x >= IM_WIDTH or y2 + y >= IM_HEIGHT:
                continue
            new_boxes.append(torch.as_tensor((x1 + x, y1 + y, x2 + x, y2 + y)).float())
        data["boxes"] = torch.stack(new_boxes)
        x_crop = min(x + imw, IM_WIDTH) - x
        y_crop = min(y + imh, IM_HEIGHT) - y
        img = np.asarray(img, dtype=np.uint8)
        img_full[y:y+imh, x:x+imw, :3] = img[:y_crop, :x_crop, :]
        for i, line in enumerate(img_full[y:y+imh]):
            for j, cell in enumerate(line[x:x+imw]):
                if (img_full[y+i, x+j, :3] != 0).any():
                    img_full[y+i, x+j, 3] = 255

        img_full = torch.tensor(img_full)
        img_full = img_full.permute(2, 0, 1)
        img_full, data = transforms(img_full, data)
        img_full = T.ToPILImage()(img_full)
        card_masks.append((img_full, data))

    # resize background
    bg_image = bg_image.resize((IM_WIDTH, IM_HEIGHT))

    final_targets = {"boxes": [], "labels": []}
    # start stacking the photos - always check if a new image obscures the labels of an old one
    for idx, (img, data) in enumerate(card_masks):
        bg_image.paste(img, (0, 0), img)
        bad_boxes = set()
        for box in final_targets["boxes"]:
            x1, y1, x2, y2 = map(int, box.numpy())
            img_arr = np.asarray(img)
            box_on_mask = img_arr[y1:y2, x1:x2, 3]
            box_on_mask = box_on_mask[box_on_mask != 0]
            if len(box_on_mask) > 0:
                bad_boxes.add((x1, y1, x2, y2))
        for bad_box in bad_boxes:
            idx = None
            for id, box in enumerate(final_targets["boxes"]):
                x1, y1, x2, y2 = map(int, box.numpy())
                if (x1, y1, x2, y2) == bad_box:
                    idx = id
                    break
            del final_targets["boxes"][idx]
            del final_targets["labels"][idx]
        final_targets["boxes"].extend(data["boxes"])
        final_targets["labels"].extend(data["labels"])
    bg_image.show()
    final_targets["boxes"] = torch.stack(final_targets["boxes"])
    final_targets["labels"] = torch.stack(final_targets["labels"])
    return bg_image, final_targets

MAX_IM_SIZE = 300
def get_random_img(dir_path: str) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    fps = glob.glob(os.path.join(dir_path, "*.jpg"))
    xmls = glob.glob(os.path.join(dir_path, "*.xml"))

    idx = np.random.randint(low=0, high=len(fps))
    img = Image.open(fps[idx])
    targets = parse_xml(xmls[idx])

    max_dim = np.max(img.size)
    w, h = img.size
    scale_factor = MAX_IM_SIZE / max_dim
    img = img.resize((int(w * scale_factor), int(h * scale_factor)))
    data_dict = {"boxes": [], "labels": []}
    for cls, *box in targets:
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_factor)
        y1 = int(y1 * scale_factor)
        x2 = int(x2 * scale_factor)
        y2 = int(y2 * scale_factor)
        data_dict["boxes"].append(torch.tensor((x1, y1, x2, y2)))
        data_dict["labels"].append(torch.tensor([possible_classes[cls]]))
    data_dict["boxes"] = torch.stack(data_dict["boxes"])
    data_dict["labels"] = torch.stack(data_dict["labels"])
    return img, data_dict

if __name__ == "__main__":
    bg_image = get_random_bg_img(get_image_files())
    # img, targets = generate_random_image()
    imgs_data = []
    for i in range(5):
        imgs_data.append(get_random_img("D:\\facultate stuff\\licenta\\data\\RAW\\my-stuff-cropped"))
    res, res_data = generate_random_image(*imgs_data, bg_image=bg_image)
    res = np.array(res)
    res = torch.from_numpy(res)
    res = res.permute(2, 0, 1)
    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    drawn_boxes = draw_bounding_boxes(transforms.ConvertImageDtype(torch.uint8)(res), res_data["boxes"],
                                      colors="red")
    show(drawn_boxes)
    plt.show()