from typing import List, Optional, Dict, Tuple
import glob
import os
import random

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from Utils.file_utils import get_image_files, get_random_bg_img, get_random_img, load_img_and_xml, MAX_IM_SIZE
from Utils.trans import RandomAffineBoxSensitive, RandomPerspectiveBoxSensitive, MyCompose, RandomColorJitterBoxSensitive, RandomGaussianNoise, \
                        RandomStretch
from Data_Processing.Raw_Train_Data.raw import parse_xml, possible_classes, pos_cls_inverse
"""
Generate a dataset of a certain size from a given dataset of cropped cards and another dataset of random backgrounds.

"""

IM_HEIGHT = 1080//1.75
IM_WIDTH = 1900//1.75
SQ_2 = 1.414


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
       (#RandomGaussianNoise(mean=0., var=0.07, prob=0.9),
        RandomColorJitterBoxSensitive(brightness=0.7, prob=0.7),
        RandomAffineBoxSensitive(degrees=(0, 350), scale=(0.9, 1.1), prob=0.6),
        RandomPerspectiveBoxSensitive(dist_scale=0.5, prob=0.3))
    )
    transforms_digital = MyCompose(
       (#RandomGaussianNoise(mean=0., var=1e-6, prob=0.9),
        RandomColorJitterBoxSensitive(brightness=0.7, prob=0.7),
        RandomAffineBoxSensitive(degrees=(0, 350), scale=(0.9, 1.1), prob=0.6),
        RandomPerspectiveBoxSensitive(dist_scale=0.5, prob=0.3))
    )

    # create the card masks, they will be a couple of black images with cards on them after various transforms
    for (img, data) in cards:
        img_full = np.zeros((IM_HEIGHT, IM_WIDTH, 4), dtype=np.uint8)

        PAD_SIZE = int(MAX_IM_SIZE * SQ_2 * 1.5)
        # use a padding array to avoid applying the transforms on the whole image
        padding = np.zeros((PAD_SIZE, PAD_SIZE, 4), dtype=np.uint8)
        imw, imh = img.size
        img = np.asarray(img, dtype=np.uint8)
        padding[(PAD_SIZE-imh)//2:(PAD_SIZE-imh)//2+imh,
        (PAD_SIZE-imw)//2:(PAD_SIZE-imw)//2+imw, :3] = img[:, :, :3]
        # create image mask
        for i, line in enumerate(padding[(PAD_SIZE-imh)//2:(PAD_SIZE-imh)//2+imh]):
            for j, cell in enumerate(line[(PAD_SIZE-imw)//2:(PAD_SIZE-imw)//2+imw]):
                padding[(PAD_SIZE-imh)//2+i, (PAD_SIZE-imw)//2+j, 3] = 255
        boxes = data["boxes"]
        # recalculate bounding boxes positions
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            data["boxes"][idx] = torch.as_tensor([x1+(PAD_SIZE-imw)//2, y1+(PAD_SIZE-imh)//2,
                                                  x2+(PAD_SIZE-imw)//2, y2+(PAD_SIZE-imh)//2])
        data["boxes"] = data["boxes"].float()
        padding = torch.from_numpy(padding)
        padding = padding.permute(2, 0, 1)
        if possible_classes["JOKER_red"] in data["labels"]:
            padding, data = transforms_digital(padding.to("cuda"), data)  # apply transforms on cuda
        else:
            padding, data = transforms(padding.to("cuda"), data) # apply transforms on cuda
        padding = padding.cpu()
        padding = padding.permute(1, 2, 0)


        boxes = data["boxes"]   
        new_boxes = []
        good_idx = []
        h, w, *_ = padding.shape
        x = torch.randint(low=20, high=IM_WIDTH - w//2, size=(1,)).item()
        y = torch.randint(low=20, high=IM_HEIGHT - h//2, size=(1,)).item()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 + x >= IM_WIDTH or y1 + y >= IM_HEIGHT or x2 + x - IM_WIDTH >= IM_WIDTH - x1 - x or \
                    y2 + y - IM_HEIGHT >= IM_HEIGHT - y1 - y:
                continue
            x1 = max(x1 + x, 0)
            y1 = max(y1 + y, 0)
            x2 = min(x2 + x, IM_WIDTH - 1)
            y2 = min(y2 + y, IM_HEIGHT - 1)
            new_boxes.append(torch.as_tensor((x1, y1, x2, y2)).float())
            good_idx.append(idx)
        if len(new_boxes) > 0:
            data["boxes"] = torch.stack(new_boxes)
        else:
            data["boxes"] = torch.as_tensor([])
        data["labels"] = data["labels"][good_idx]
        # for the case in which the entire padding image can't fully fit in the image
        x_crop = min(x + w, IM_WIDTH) - x
        y_crop = min(y + h, IM_HEIGHT) - y
        img_full[y:y+y_crop, x:x+x_crop, :] = padding[:y_crop, :x_crop, :]
        img_full = Image.fromarray(img_full)

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
            if len(box_on_mask) / ((x2 - x1) * (y2 - y1)) > 0.3:
                bad_boxes.append(idx)
        bad_boxes = np.asarray(bad_boxes)
        # remove obscured boxes from the scene
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


def choose_cut_on_prev(prev_cut: str) -> str:
    if prev_cut == "v1":
        return random.choices(["h1", "h2", "d3", "d4"], weights=[0.15, 0.15, 0.35, 0.35])[0]
    elif prev_cut == "v2":
        return random.choices(["h1", "h2", "d1", "d2"], weights=[0.15, 0.15, 0.35, 0.35])[0]
    elif prev_cut == "h1":
        return random.choice(["v1", "v2", "d4", "d1"])
    elif prev_cut == "h2":
        return random.choice(["v1", "v2", "d3", "d2"])
    elif prev_cut == "d1":
        return random.choice(["h1", "v2"])
    elif prev_cut == "d2":
        return random.choice(["v2", "h2"])
    elif prev_cut == "d3":
        return random.choice(["v1", "h2"])
    elif prev_cut == "d4":
        return random.choice(["v1", "h1"])


def transform_coords(diag: str, a: float, x: int, side_len: int) -> float:
    if diag == "d1":
        return (x/side_len) / (2*(1-a)) - a/(2*(1-a))
    elif diag == "d2":
        return (x/side_len) / (2*(a-1)) + (a-2)/(2*(a-1))
    elif diag == "d3":
        return (x/side_len)/(2*a) + 1/2
    elif diag == "d4":
        return -(x/side_len)/(2*a) + 1/2


def perform_cut(full_img, cut: str, bbox: List[int]):
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.int64)
    x3, y3, x4, y4 = x2, y1, x1, y2
    if cut == "v1" or cut == "v2":
        # v1 is cutting a portion of the left part of the bbox
        cut_fraction = np.random.uniform(0.2, 0.4)
        xstart = int((cut == "v1") * 0 + (cut == "v2") * ((x2 - x1) - cut_fraction * (x2 - x1))) + x1
        xstop = int((cut == "v1") * (cut_fraction * (x2 - x1)) + (cut == "v2") * (x2 - x1)) + x1
        full_img[y1:y2, xstart:xstop, :3] = np.random.randint(0, 256) # remove masks
    elif cut == "h1" or cut == "h2":
        cut_fraction = 0.2
        ystart = int((cut == "h1") * 0 + (cut == "h2") * ((y2 - y1) - cut_fraction * (y2 - y1))) + y1
        ystop = int((cut == "h1") * (cut_fraction * (y2 - y1)) + (cut == "h2") * (y2 - y1)) + y1
        full_img[ystart:ystop, x1:x2, :3] = np.random.randint(0, 256) # remove masks
    else:
        # diagonal cuts, might be a huge bottleneck
        a = np.random.uniform(0.25, 0.5) # coefficient for x axis
        if cut == "d1" or cut == "d2":
            a = 1 - a
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                good_x = ((cut == "d1" or cut == "d2") and x >= a * (x2 - x1) + x1) or\
                         ((cut == "d3" or cut == "d4") and x <= a * (x2 - x1) + x1)
                if not good_x:
                    continue
                good_y = (int(transform_coords(cut, a, x-x1, x2 - x1) * (y2 - y1) + y1) >= y and (cut == "d1" or cut == "d4")) or\
                         (int(transform_coords(cut, a, x-x1, x2 - x1) * (y2 - y1) + y1) <= y and (cut == "d2" or cut == "d3"))
                if not good_y:
                    continue
                full_img[y, x, :3] = np.random.randint(0, 256) # remove masks
    return full_img


def generate_handlike_image(*cards, bg_image: Image.Image) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
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
       (#RandomGaussianNoise(mean=0., var=0.05, prob=1.),
        RandomColorJitterBoxSensitive(brightness=0.7, prob=0.7),
        RandomAffineBoxSensitive(degrees=(0, 350), scale=(0.5, 1.5), prob=0.6),
        RandomPerspectiveBoxSensitive(dist_scale=0.5, prob=0.3))
    )
    # create the card masks, they will be a couple of black images with cards on them after various transforms
    for (img, data) in cards:
        img_full = np.zeros((IM_HEIGHT, IM_WIDTH, 4), dtype=np.uint8)

        PAD_SIZE = int(MAX_IM_SIZE * SQ_2 * 1.5)
        # use a padding array to avoid applying the transforms on the whole image
        padding = np.zeros((PAD_SIZE, PAD_SIZE, 4), dtype=np.uint8)
        imw, imh = img.size
        img = np.asarray(img, dtype=np.uint8)
        padding[(PAD_SIZE-imh)//2:(PAD_SIZE-imh)//2+imh,
        (PAD_SIZE-imw)//2:(PAD_SIZE-imw)//2+imw, :3] = img[:, :, :3]
        # create image mask
        for i, line in enumerate(padding[(PAD_SIZE-imh)//2:(PAD_SIZE-imh)//2+imh]):
            for j, cell in enumerate(line[(PAD_SIZE-imw)//2:(PAD_SIZE-imw)//2+imw]):
                padding[(PAD_SIZE-imh)//2+i, (PAD_SIZE-imw)//2+j, 3] = 255
        boxes = data["boxes"]
        # recalculate bounding boxes positions
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            data["boxes"][idx] = torch.as_tensor([x1+(PAD_SIZE-imw)//2, y1+(PAD_SIZE-imh)//2,
                                                  x2+(PAD_SIZE-imw)//2, y2+(PAD_SIZE-imh)//2])
        data["boxes"] = data["boxes"].float()
        padding = torch.from_numpy(padding)
        padding = padding.permute(2, 0, 1)
        padding, data = transforms(padding.to("cuda"), data) # apply transforms on cuda
        padding = padding.cpu()
        padding = padding.permute(1, 2, 0)


        boxes = data["boxes"]
        new_boxes = []
        good_idx = []
        x = torch.randint(low=20, high=IM_WIDTH - PAD_SIZE//2, size=(1,)).item()
        y = torch.randint(low=20, high=IM_HEIGHT - PAD_SIZE//2, size=(1,)).item()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 + x >= IM_WIDTH or y1 + y >= IM_HEIGHT or x2 + x - IM_WIDTH >= 1/2 * (IM_WIDTH - (x1 + x)) or \
                    y2 + y - IM_HEIGHT >= 1/2*(IM_HEIGHT - (y1 + y)):
                continue
            x1 = max(x1 + x, 0)
            y1 = max(y1 + y, 0)
            x2 = min(x2 + x, IM_WIDTH - 1)
            y2 = min(y2 + y, IM_HEIGHT - 1)
            new_boxes.append(torch.as_tensor((x1, y1, x2, y2)).float())
            good_idx.append(idx)
        if len(new_boxes) > 0:
            data["boxes"] = torch.stack(new_boxes)
        else:
            data["boxes"] = torch.as_tensor([])
        data["labels"] = data["labels"][good_idx]
        # for the case in which the entire padding image can't fully fit in the image
        x_crop = min(x + PAD_SIZE, IM_WIDTH) - x
        y_crop = min(y + PAD_SIZE, IM_HEIGHT) - y
        img_full[y:y+PAD_SIZE, x:x+PAD_SIZE, :] = padding[:y_crop, :x_crop, :]
        img_full = Image.fromarray(img_full)

        card_masks.append((img_full, data))

    # apply cuts on bboxes
    for idx, (img, data) in enumerate(card_masks):
        boxes = data["boxes"]
        img = np.asarray(img)
        for box in boxes:
            x1, y1, x2, y2 = box
            x3, y3, x4, y4 = x2, y1, x1, y2
            # choose how many cuts to apply to the label box
            cuts_no = np.random.randint(low=1, high=3)
            cuts_types = [("v1", "v2"), ("h1", "h2"), ("d1", "d2", "d3", "d4")]
            prev_cut = None
            while cuts_no > 0:
                cut = cuts_types[np.random.choice(len(cuts_types))]
                cuts_types.remove(cut)
                if prev_cut is None:
                    # for the first cut, just choose it randomly
                    cut = cut[np.random.choice(len(cut))]
                    prev_cut = cut
                else:
                    cut = choose_cut_on_prev(prev_cut)
                img = perform_cut(img, cut, box)
                cuts_no -= 1
        img = Image.fromarray(img)
        card_masks[idx] = (img, data)

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
            if len(box_on_mask) / ((x2 - x1) * (y2 - y1)) > 0.3:
                bad_boxes.append(idx)
        bad_boxes = np.asarray(bad_boxes)
        # remove obscured boxes from the scene
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
            if label.item() == -1:
                label[0] = 0
            cls_name = pos_cls_inverse[label.item()]
            # for cls, labs in possible_classes.items():
            #     if labs == label:
            #         cls_name = cls
            f.write(xml_obj.format(CLASS=cls_name, XMIN=x1, YMIN=y1, XMAX=x2, YMAX=y2))
        f.write(xml_end)


def gen_dataset_from_dir(root_dir: str, dest_dir: str, num_datasets: int, prob_datasets: List[float],
                         cards_to_ignore: List[int]=None, start_from: int=0, num_imgs: int=10**4, stretch: bool=False) -> None:
    """
    Generate a dataset using images from root dir and writing them to dest_dir. It is assumed a dataset contains all 54
    classes, with the possible exception of red Joker.
    :param root_dir: root dir of dataset(s)
    :param dest_dir: dest dir of generated dataset
    :param num_datasets: number of different datasets in root dir
    :param prob_datasets: probability of choosing a card from certain dataset
    :param cards_to_ignore: cards to ignore in generation, useful for balancing datasets
    :param start_from: where to start from the naming, useful for not rewriting already generated images
    :param num_imgs: how many images to generate
    :return: Nothing, the function will write the images and their annotations to the dest dir
    """
    assert len(prob_datasets) == num_datasets
    # uniform distribution if not given any
    if prob_datasets is None:
        prob_datasets = [1/num_datasets for i in range(num_datasets)]
    datasets = [i for i in range(num_datasets)]
    bg_file_dir = get_image_files()


    if cards_to_ignore is None:
        card_classes = [i for i in range(1, 55)]
        card_bins = [1 for i in range(1, 55)]
        card_total = 54
        card_prob = [1/54 for i in range(1, 55)]
    else:
        card_classes = [i for i in range(1, 55) if i not in cards_to_ignore]
        card_bins = [1 for i in range(len(card_classes))]
        card_total = len(card_classes)
        card_prob = [1/len(card_classes) for i in range(len(card_classes))]

    for i in tqdm(range(num_imgs)):
        num_cards = np.random.randint(low=2, high=6)

        cards = []
        # get the cards
        for ci in range(num_cards):
            card_prb, card_cls = zip(*sorted(zip(card_prob, card_classes)))
            card_cls = list(card_cls)
            card_cls.reverse()
            card = np.random.choice(card_cls, p=card_prb)
            dataset = np.random.choice(datasets, p=prob_datasets)
            if dataset == 0:
                dataset = ""
            else:
                dataset = str(dataset+1)

            if card == possible_classes["JOKER_red"] and not os.path.exists(os.path.join(root_dir, "joker_red" + dataset
                                                                                         + ".jpg")):
                card = possible_classes["JOKER_black"]

            img, targets = load_img_and_xml(root_dir, pos_cls_inverse[card].lower() + dataset, stretched=stretch)
            cards.append((img, targets))

        # get the background image
        bg_image = get_random_bg_img(bg_file_dir)

        # generate the new image
        img, targets = generate_random_image(*cards, bg_image=bg_image)
        for card in targets["labels"]:
            card = card.item()
            card_index = card_classes.index(card) # needed only for when balancing datasets
            # card_index = card - 1 # the regular way to go about it
            card_bins[card_index] += 1
            card_total += 1
        # use this piece of code to ignore red jokers for datasets that lack them
        # also keep in mind it won't work for balancing your dataset
        # card_total -= card_bins[possible_classes["JOKER_red"] - 1]
        # card_bins[possible_classes["JOKER_red"] - 1] = card_bins[possible_classes["JOKER_black"] - 1]
        # card_total += card_bins[possible_classes["JOKER_black"] - 1]
        card_prob = [card_val/card_total for card_val in card_bins] # update probabilities
        write_annotation(dest_dir, str(start_from + i), img.size[::-1], targets)
        img.save(os.path.join(dest_dir, str(start_from + i) + ".jpg"))

def gen_dataset_handlike_from_dir(root_dir: str, dest_dir: str, num_datasets: int, prob_datasets: List[float],
                         cards_to_ignore: List[int]=None, start_from: int=0, num_imgs: int=10**4) -> None:
    """
    Generate a dataset using images from root dir and writing them to dest_dir. It is assumed a dataset contains all 54
    classes, with the possible exception of red Joker. The resulting images will have a "handlike" distribution.
    :param root_dir: root dir of dataset(s)
    :param dest_dir: dest dir of generated dataset
    :param num_datasets: number of different datasets in root dir
    :param prob_datasets: probability of choosing a card from certain dataset
    :param cards_to_ignore: cards to ignore in generation, useful for balancing datasets
    :param start_from: where to start from the naming, useful for not rewriting already generated images
    :param num_imgs: how many images to generate
    :return: Nothing, the function will write the images and their annotations to the dest dir
    """
    assert len(prob_datasets) == num_datasets
    # uniform distribution if not given any
    if prob_datasets is None:
        prob_datasets = [1/num_datasets for i in range(num_datasets)]
    datasets = [i for i in range(num_datasets)]
    bg_file_dir = get_image_files()


    if cards_to_ignore is None:
        card_classes = [i for i in range(1, 55)]
        card_bins = [1 for i in range(1, 55)]
        card_total = 54
        card_prob = [1/54 for i in range(1, 55)]
    else:
        card_classes = [i for i in range(1, 55) if i not in cards_to_ignore]
        card_bins = [1 for i in range(len(card_classes))]
        card_total = len(card_classes)
        card_prob = [1/len(card_classes) for i in range(len(card_classes))]

    for i in tqdm(range(num_imgs)):
        num_cards = np.random.randint(low=2, high=6)

        cards = []
        # get the cards
        for ci in range(num_cards):
            card_prb, card_cls = zip(*sorted(zip(card_prob, card_classes)))
            card_cls = list(card_cls)
            card_cls.reverse()
            card = np.random.choice(card_cls, p=card_prb)
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
        img, targets = generate_handlike_image(*cards, bg_image=bg_image)
        for card in targets["labels"]:
            card = card.item()
            if card == -1:
                continue
            card_index = card_classes.index(card) # needed only for when balancing datasets
            # card_index = card - 1 # the regular way to go about it
            card_bins[card_index] += 1
            card_total += 1
        # use this piece of code to ignore red jokers for datasets that lack them
        # also keep in mind it won't work for balancing your dataset
        # card_total -= card_bins[possible_classes["JOKER_red"] - 1]
        # card_bins[possible_classes["JOKER_red"] - 1] = card_bins[possible_classes["JOKER_black"] - 1]
        # card_total += card_bins[possible_classes["JOKER_black"] - 1]
        card_prob = [card_val/card_total for card_val in card_bins] # update probabilities
        write_annotation(dest_dir, str(start_from + i), img.size[::-1], targets)
        img.save(os.path.join(dest_dir, str(start_from + i) + ".jpg"))


def dataset_statistics(dataset_path: str) -> None:
    files = glob.glob(os.path.join(dataset_path, "*.xml"))
    final_results = {}

    for file in tqdm(files):
        xml_stuff = parse_xml(file)
        for cls, *box in xml_stuff:
            if cls in final_results:
                final_results[cls] += 1
            else:
                final_results[cls] = 1
    for cls in possible_classes:
        if cls not in final_results:
            continue
        print(f"Class {cls} has {final_results[cls]} examples.")
    sum = 0
    for cls in final_results:
        sum += final_results[cls]
    mean = sum / (len(final_results))
    sum = 0
    for cls in final_results:
        sum += (final_results[cls] - mean) ** 2
    variance = sum / (len(final_results))
    print(f"Mean of dataset is {mean}.")
    print(f"Variance of dataset is {variance}.")

    plt.bar([key for key in final_results], [val for key, val in final_results.items()])
    plt.title(f"Data labels spread, mean of {mean} and variance of {variance}")
    plt.show()

def resize_dataset(root_dir: str, dest_dir: str, res_factor: float) -> None:
    files = glob.glob(os.path.join(root_dir, "*.jpg"))
    for i in tqdm(range(len(files))):
        file = files[i]
        img = Image.open(file)
        xml = parse_xml(file[:-3] + "xml")

        max_dim = np.max(img.size)
        w, h = img.size
        img = img.resize((int(w * res_factor), int(h * res_factor)))

        data_dict = {"boxes": [], "labels": []}
        for cls, *box in xml:
            x1, y1, x2, y2 = box
            x1 = int(x1 * res_factor)
            y1 = int(y1 * res_factor)
            x2 = int(x2 * res_factor)
            y2 = int(y2 * res_factor)
            data_dict["boxes"].append(torch.tensor((x1, y1, x2, y2)))
            data_dict["labels"].append(torch.tensor([possible_classes[cls]]))
        if len(data_dict["boxes"]) > 0:
            data_dict["boxes"] = torch.stack(data_dict["boxes"])
            data_dict["labels"] = torch.stack(data_dict["labels"])
        img.save(os.path.join(dest_dir, str(i) + ".jpg"))
        write_annotation(dest_dir, str(i), img.size[::-1], data_dict)

if __name__ == "__main__":
    # dataset_statistics("../../data/my_stuff_augm/testing/")
    # resize_dataset("../../data/RAW/my-stuff-cropped/", "../../data/RAW/my-stuff-cropped-res/", 0.3)
    gen_dataset_from_dir("../../data/RAW/PNG-cards-1.3/PNG-cards-1.3/", "../../data/my_stuff_augm/stretched/", num_datasets=1,
                         prob_datasets=[1.], num_imgs=1 * 10**4, start_from=11 * 10 **4, stretch=True)