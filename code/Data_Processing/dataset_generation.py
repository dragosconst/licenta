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

from Utils.file_utils import get_image_files, get_random_bg_img, get_random_img, load_img_and_xml
from Utils.trans import RandomAffineBoxSensitive, RandomPerspectiveBoxSensitive, MyCompose, RandomColorJitterBoxSensitive
from Data_Processing.Raw_Train_Data.raw import parse_xml, possible_classes, pos_cls_inverse
"""
Generate a dataset of a certain size from a given dataset of cropped cards and another dataset of random backgrounds.

"""

IM_HEIGHT = 1080
IM_WIDTH = 1900
def generate_random_image(*cards, bg_image: Image.Image) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    """
    Generate a random image with given cards and bg_image. Note that the cards should ALREADY be resized prior to calling
    this function.
    :param cards: tuples of (img, data), where img is the pic of the card and data represents the relevant labeling data
    :param bg_image: a random background image
    :return: tuple of image and data dictionary with labels and bboxes
    """
    assert cards is not None
    assert len(cards) != 0

    card_masks = []
    transforms = MyCompose(
       (RandomColorJitterBoxSensitive(brightness=0.6, prob=0.4),
        RandomAffineBoxSensitive(degrees=(0, 350), scale=(0.5, 1.5), prob=0.6),
        RandomPerspectiveBoxSensitive(dist_scale=0.4, prob=0.3))
    )
    # create the card masks, they will be a couple of black images with cards on them after various transforms
    for (img, data) in cards:
        img_full = np.zeros((IM_HEIGHT, IM_WIDTH, 4), dtype=np.uint8)
        imw, imh = img.size
        x = torch.randint(low=20, high=IM_WIDTH - imw//2, size=(1,)).item()
        y = torch.randint(low=20, high=IM_HEIGHT - imh//2, size=(1,)).item()
        boxes = data["boxes"]
        new_boxes = []
        good_idx = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 + x >= IM_WIDTH or y1 + y >= IM_HEIGHT or x2 + x >= IM_WIDTH or y2 + y >= IM_HEIGHT:
                continue
            new_boxes.append(torch.as_tensor((x1 + x, y1 + y, x2 + x, y2 + y)).float())
            good_idx.append(idx)
        data["boxes"] = torch.stack(new_boxes)
        data["labels"] = data["labels"][good_idx]
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
        bad_boxes = []
        # explictly check if this image severly obscures any label
        for idx, box in enumerate(final_targets["boxes"]):
            x1, y1, x2, y2 = map(int, box.numpy())
            img_arr = np.asarray(img)
            box_on_mask = img_arr[y1:y2, x1:x2, 3]
            box_on_mask = box_on_mask[box_on_mask != 0]
            if len(box_on_mask) > 60:
                bad_boxes.append(idx)
        for bad_box in bad_boxes[::-1]:
            final_targets["boxes"].pop(bad_box)
            final_targets["labels"].pop(bad_box)
        final_targets["boxes"].extend(data["boxes"])
        final_targets["labels"].extend(data["labels"])
    # bg_image.show()
    if len(final_targets["boxes"]) > 0:
        final_targets["boxes"] = torch.stack(final_targets["boxes"])
        final_targets["labels"] = torch.stack(final_targets["labels"])
    else:
        final_targets["boxes"] = torch.Tensor([])
        final_targets["labels"] = torch.Tensor([])
    return bg_image, final_targets


def write_annotation(fp: str, name: str, shape: Tuple[int, int], data: Dict[str, torch.Tensor]) -> None:
    """
    Write data dictionary to output xml file
    :param fp: filepath of dest directory
    :param name: filename of parent jpg
    :param shape: shape of image in (h, w) format
    :param data: data dictionary to read from
    :return: Nothing
    """
    xml_head = """<annotation>
            <folder>images_created</folder>
            <filename>{FILENAME}</filename>
            <path>{PATH}</path>
            <source>
                    <database>Unknown</database>
            </source>
            <size>
                    <width>{WIDTH}</width>
                    <height>{HEIGHT}</height>
                    <depth>3</depth>
            </size>
    """
    xml_obj = """ <object>
                    <name>{CLASS}</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                            <xmin>{XMIN}</xmin>
                            <ymin>{YMIN}</ymin>
                            <xmax>{XMAX}</xmax>
                            <ymax>{YMAX}</ymax>
                    </bndbox>
            </object>
    """
    xml_end = """</annotation>        
    """
    xml_file = os.path.join(fp, name + ".xml")
    with open(xml_file, "w+") as f:
        f.write(xml_head.format(FILENAME=name + ".jpg", PATH=os.path.join(fp, name + ".jpg"),
                                WIDTH=shape[1], HEIGHT=shape[0]))
        for box, label in zip(data["boxes"], data["labels"]):
            x1, y1, x2, y2 = list(map(int, box))
            cls_name = pos_cls_inverse[label.item()]
            # for cls, labs in possible_classes.items():
            #     if labs == label:
            #         cls_name = cls
            f.write(xml_obj.format(CLASS=cls_name, XMIN=x1, YMIN=y1, XMAX=x2, YMAX=y2))
        f.write(xml_end)


def gen_dataset_from_dir(root_dir: str, dest_dir: str, num_datasets: int, prob_datasets: List[float],
                         num_imgs: int=10**4):
    """
    Generate a dataset using images from root dir and writing them to dest_dir. It is assumed a dataset contains all 54
    classes, with the possible exception of red Joker.
    :param root_dir: root dir of dataset(s)
    :param dest_dir: dest dir of generated dataset
    :param num_datasets: number of different datasets in root dir
    :param prob_datasets: probability of choosing a card from certain dataset
    :param num_imgs: how many images to generate
    :return: Nothing, the function will write the images and their annotations to the dest dir
    """
    assert len(prob_datasets) == num_datasets
    # uniform distribution if not given any
    if prob_datasets is None:
        prob_datasets = [1/num_datasets for i in range(num_datasets)]
    datasets = [i for i in range(num_datasets)]
    bg_file_dir = get_image_files()

    for i in tqdm(range(num_imgs)):
        num_cards = np.random.randint(low=2, high=6)

        cards = []
        # get the cards
        for ci in range(num_cards):
            card = np.random.randint(low=1, high=55)
            dataset = np.random.choice(datasets, p=prob_datasets)
            if dataset == 0:
                dataset = ""
            else:
                dataset = str(dataset+1)

            if card == possible_classes["JOKER_red"] and not os.path.exists(os.path.join(root_dir, "joker_red" + dataset
                                                                                         + ".jpg")):
                card = possible_classes["JOKER_black"]

            img, targets = load_img_and_xml(root_dir, pos_cls_inverse[card].lower() + dataset)
            cards.append((img, targets))

        # get the background image
        bg_image = get_random_bg_img(bg_file_dir)

        # generate the new image
        img, targets = generate_random_image(*cards, bg_image=bg_image)
        # img = np.array(img)
        # img = torch.from_numpy(img)
        # img = img.permute(2, 0, 1) # i have to do this weird incantantion because torch won't recognize the PIL image by itself for some reason
        write_annotation(dest_dir, str(i), img.size[::-1], targets)
        img.save(os.path.join(dest_dir, str(i) + ".jpg"))



if __name__ == "__main__":
    gen_dataset_from_dir("../../data/RAW/my-stuff-cropped/", "../../data/my_stuff_augm/", num_datasets=2,
                         prob_datasets=[0.7, 0.3], num_imgs=100)
    # bg_image = get_random_bg_img(get_image_files())
    # # img, targets = generate_random_image()
    # imgs_data = []
    # for i in range(5):
    #     imgs_data.append(get_random_img("D:\\facultate stuff\\licenta\\data\\RAW\\my-stuff-cropped"))
    # res, res_data = generate_random_image(*imgs_data, bg_image=bg_image)
    # res = np.array(res)
    # res = torch.from_numpy(res)
    # res = res.permute(2, 0, 1)
    # def show(imgs):
    #     if not isinstance(imgs, list):
    #         imgs = [imgs]
    #     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    #     for i, img in enumerate(imgs):
    #         img = img.detach()
    #         img = F.to_pil_image(img)
    #         axs[0, i].imshow(np.asarray(img))
    #         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # print(len(res_data["boxes"]))
    # drawn_boxes = draw_bounding_boxes(transforms.ConvertImageDtype(torch.uint8)(res), res_data["boxes"],
    #                                   colors="red")
    # show(drawn_boxes)
    # write_annotation(os.getcwd(), "ahaha", res.size()[1:], res_data)
    # # transforms.ToPILImage()(res).save("ahaha.jpg")
    # plt.show()