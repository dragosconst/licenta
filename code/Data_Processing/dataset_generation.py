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
import matplotlib.pyplot as plt

from Utils.file_utils import get_image_files, get_random_bg_img, get_random_img, load_img_and_xml, MAX_IM_SIZE
from Utils.trans import RandomAffineBoxSensitive, RandomPerspectiveBoxSensitive, MyCompose, RandomColorJitterBoxSensitive, RandomGaussianNoise
from Data_Processing.Raw_Train_Data.raw import parse_xml, possible_classes, pos_cls_inverse
"""
Generate a dataset of a certain size from a given dataset of cropped cards and another dataset of random backgrounds.

"""

IM_HEIGHT = 1080
IM_WIDTH = 1900
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
       (RandomGaussianNoise(mean=0., var=0.05, prob=1.),
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
        (PAD_SIZE-imw)//2:(PAD_SIZE-imw)//2+imw, :3] = img
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
        x_crop = min(x + PAD_SIZE, IM_WIDTH) - x
        y_crop = min(y + PAD_SIZE, IM_HEIGHT) - y
        img_full[y:y+PAD_SIZE, x:x+PAD_SIZE, :] = padding[:y_crop, :x_crop, :]
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

def poly_inters(p1, p2) -> bool:
    """
    use separating line to find if polygons intersect
    :param p1:
    :param p2:
    :return:
    """
    for poly in (p1, p2):
        for id1, point in enumerate(poly):
            id2 = (id1 + 1) % len(poly)
            point1 = point
            point2 = poly[id2]

            normal = (point2[0].item()-point1[0].item(),point2[1].item()-point1[1].item())

            min1 = None
            max1 = None
            for point in p1:
                projection = normal[0] * point[0].item() + normal[1] * point[1].item()
                if min1 is None or projection < min1:
                    min1 = projection
                if max1 is None or projection > max1:
                    max1 = projection

            min2, max2 = None, None
            for point in p2:
                projection = normal[0] * point[0].item() + normal[1] * point[1].item()
                if min2 is None or projection < min2:
                    min2 = projection
                if max2 is None or projection > max2:
                    max2 = projection

            if max1 < min2 or max2 < min1:
                return False
    return True

def generate_handlike_image(*cards, bg_image: Image.Image) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    """
    Generate a handlike image with given cards and bg_image. Handlike refers to the fact that the cards will somewhat
    overlap each other's boxes, to get a better simulation of real environments. Note that the cards should ALREADY be
    resized prior to calling this function.
    :param cards: tuples of (img, data), where img is the pic of the card and data represents the relevant labeling data
    :param bg_image: a random background image
    :return: tuple of image and data dictionary with labels and bboxes
    """
    assert cards is not None
    assert len(cards) != 0

    card_masks = []
    transforms = MyCompose(
       (RandomGaussianNoise(mean=0., var=0.05, prob=1., debug=True),
        RandomColorJitterBoxSensitive(brightness=0.7, prob=0.7, debug=True),
        RandomAffineBoxSensitive(degrees=(0, 350), scale=(0.5, 1.5), prob=0.6, debug=True),
        RandomPerspectiveBoxSensitive(dist_scale=0.5, prob=0.3, debug=True))
    )
    # create the card masks, they will be a couple of black images with cards on them after various transforms
    # unobscured_cards: list of tuples representing the transformed bounding boxes of cards, used to check collisions
    #                   against whole current card; by assumption, they are shrinked to 50% of their original area, by
    #                   keeping the same ratio of sides as the original bbox (and the same center)
    unobscured_cards = []
    for (img, data) in cards:
        img_full = np.zeros((IM_HEIGHT, IM_WIDTH, 4), dtype=np.uint8)

        PAD_SIZE = int(MAX_IM_SIZE * SQ_2 * 1.5)
        # use a padding array to avoid applying the transforms on the whole image
        # the padding size should be chosen such that after applying any rotation, scale or perspective shift, the image
        # should still be fully enclosed by the padding image, i.e. don't lose corners or other parts of the image after
        # applying the transforms
        padding = np.zeros((PAD_SIZE, PAD_SIZE, 4), dtype=np.uint8)
        imw, imh = img.size
        img = np.asarray(img, dtype=np.uint8)
        padding[(PAD_SIZE-imh)//2:(PAD_SIZE-imh)//2+imh,
        (PAD_SIZE-imw)//2:(PAD_SIZE-imw)//2+imw, :3] = img
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
        x1, y1, x2, y2 = (PAD_SIZE-imw)//2, (PAD_SIZE-imh)//2, (PAD_SIZE-imw)//2+imw, (PAD_SIZE-imh)//2+imh
        data["boxes"]= torch.cat((data["boxes"], torch.as_tensor([(x1, y1, x2, y2)]))) # append whole card as a box
        data["labels"] = torch.cat((data["labels"], torch.as_tensor([[-1]])))  # append whole card as a box
        data["boxes"] = data["boxes"].float()
        padding = torch.from_numpy(padding)
        padding = padding.permute(2, 0, 1)
        padding, data, debug = transforms(padding.to("cuda"), data) # apply transforms on cuda
        whole_card_bbox = data["boxes"][-1]
        whole_card = debug[-1]
        padding = padding.cpu()
        padding = padding.permute(1, 2, 0)

        boxes = data["boxes"]
        new_boxes = []
        good_idx = []

        if len(unobscured_cards) == 0:
            x = torch.randint(low=20, high=IM_WIDTH - PAD_SIZE//2, size=(1,)).item()
            y = torch.randint(low=20, high=IM_HEIGHT - PAD_SIZE//2, size=(1,)).item()
        else:
            # choose random coordinates such as they are guaranteed to obscure at least one bounding box in such
            # a way that it still should be recognizable
            cs = []
            wx1, wy1, wx2, wy2 = whole_card_bbox
            wx = (wx2 - wx1)
            wy = (wy2 - wy1)
            for card in unobscured_cards:
                x1, y1, x2, y2 = card
                cs.append([(max(x1-wx,20+wx1), max(y1-wy,20+wy1)), (min(x2+wx,IM_WIDTH - PAD_SIZE//2+wx1), min(y2+wy,\
                                                            IM_HEIGHT - PAD_SIZE//2+wy1))])
            ci_index = np.random.choice(len(cs))
            ci = cs[ci_index]
            not_intersecting = True
            x1, y1, x2, y2 = unobscured_cards[ci_index]
            card_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            while not_intersecting:
                x = torch.randint(low=ci[0][0].long().item(), high=ci[1][0].long().item(), size=(1,)).item() - wx1.long().item()
                y = torch.randint(low=ci[0][1].long().item(), high=ci[1][1].long().item(), size=(1,)).item() - wy1.long().item()
                if poly_inters(card_poly, whole_card):
                    not_intersecting = False


            # remove cards obscured by current card by checking if any of their points reside in the current card's
            # bounding box
            for card in unobscured_cards:
                x1, y1, x2, y2 = card
                x3, y3, x4, y4 = x2, y1, x2, y1
                p1 = x1 >= wx1 + x and x1 <= wx2 + x and y1 >= wy1 + y and y1 <= wy2 + y
                p2 = x2 >= wx1 + x and x2 <= wx2 + x and y2 >= wy1 + y and y2 <= wy2 + y
                p3 = x3 >= wx1 + x and x3 <= wx2 + x and y3 >= wy1 + y and y3 <= wy2 + y
                p4 = x4 >= wx1 + x and x4 <= wx2 + x and y4 >= wy1 + y and y4 <= wy2 + y
                if p1 or p2 or p3 or p4:
                    unobscured_cards.remove(card)

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
            w = (x2 - x1)
            h = (y2 - y1)
            # scale area to 50% of original area, but also keep the center and sides ratio
            scale_factor = 1/SQ_2
            nw, nh = w * scale_factor, h* scale_factor
            unobscured_cards.append((x1*scale_factor + w//2 - nw//2, y1*scale_factor + h//2 - nh//2,
                                     x2*scale_factor + w//2 - nw//2, y2*scale_factor + h//2 - nh//2))
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
            if len(box_on_mask) / ((x2 - x1) * (y2 - y1)) > 0.7:
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
                         cards_to_ignore: List[int]=None, start_from: int=0, num_imgs: int=10**4) -> None:
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

            img, targets = load_img_and_xml(root_dir, pos_cls_inverse[card].lower() + dataset)
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

    for file in tqdm(files[9000:]):
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
        if cls=="JOKER_red":
            continue
        sum += final_results[cls]
    mean = sum / (len(final_results)-1)
    sum = 0
    for cls in final_results:
        if cls=="JOKER_red":
            continue
        sum += (final_results[cls] - mean) ** 2
    variance = sum / (len(final_results)-1)
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
    # dataset_statistics("../../data/my_stuff_augm/")
    # resize_dataset("../../data/RAW/my-stuff-cropped/", "../../data/RAW/my-stuff-cropped-res/", 0.3)
    gen_dataset_handlike_from_dir("../../data/RAW/my-stuff-cropped-res/", "../../data/my_stuff_augm/testing/", num_datasets=2,
                         prob_datasets=[0.7, 0.3], num_imgs=10, start_from=1)