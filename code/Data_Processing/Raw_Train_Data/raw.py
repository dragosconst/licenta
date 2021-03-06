import os
import glob
import pdb
from functools import cmp_to_key
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import cv2 as cv
from tqdm import tqdm

from global_constants import CARD_WIDTH, CARD_HEIGHT

"""
This script was used to handle sorting through the unmodified datasets I've gathered from the internet or from
my own photos.
"""

possible_classes = {"Ac": 1, "Ad": 2, "Ah": 3, "As": 4,
                    "2c": 5, "2d": 6, "2h": 7, "2s": 8,
                    "3c": 9, "3d": 10, "3h": 11, "3s": 12,
                    "4c": 13, "4d": 14, "4h": 15, "4s": 16,
                    "5c": 17, "5d": 18, "5h": 19, "5s": 20,
                    "6c": 21, "6d": 22, "6h": 23, "6s": 24,
                    "7c": 25, "7d": 26, "7h": 27, "7s": 28,
                    "8c": 29, "8d": 30, "8h": 31, "8s": 32,
                    "9c": 33, "9d": 34, "9h": 35, "9s": 36,
                    "10c": 37, "10d": 38, "10h": 39, "10s": 40,
                    "Jc": 41, "Jd": 42, "Jh": 43, "Js": 44,
                    "Qc": 45, "Qd": 46, "Qh": 47, "Qs": 48,
                    "Kc": 49, "Kd": 50, "Kh": 51, "Ks": 52,
                    "JOKER_red": 53, "JOKER_black": 54,
                    "not a card": 0
                    } # set of all possible classes

pos_cls_inverse = { 1: "Ac", 2: "Ad", 3: "Ah", 4: "As",
                    5: "2c", 6: "2d", 7: "2h", 8: "2s",
                    9: "3c", 10: "3d", 11: "3h", 12: "3s",
                    13: "4c", 14: "4d", 15: "4h", 16: "4s",
                    17: "5c", 18: "5d", 19: "5h", 20: "5s",
                    21: "6c", 22: "6d", 23: "6h", 24: "6s",
                    25: "7c", 26: "7d", 27: "7h", 28: "7s",
                    29: "8c", 30: "8d", 31: "8h", 32: "8s",
                    33: "9c", 34: "9d", 35: "9h", 36: "9s",
                    37: "10c", 38: "10d", 39: "10h", 40: "10s",
                    41: "Jc", 42: "Jd", 43: "Jh", 44: "Js",
                    45: "Qc", 46: "Qd", 47: "Qh", 48: "Qs",
                    49: "Kc", 50: "Kd", 51: "Kh", 52: "Ks",
                    53: "JOKER_red", 54: "JOKER_black",
                    0: "not a card"
} # inverse set of classes mappings and names in labeling, useful for fast operations

all_classes = {
                "Ac": ["Ac", "AC", "ace of clubs"], "Ad": ["Ad", "AD", "ace of diamonds"],
                "Ah": ["Ah", "AH", "ace of hearts"], "As": ["As", "AS", "ace of spades"],
                "1c": ["1c", "1C", "one of clubs"], "1d": ["1d", "1D", "one of diamonds"],
                "1h": ["1h", "1H", "one of hearts"], "1s": ["1s", "1S", "one of spades"],
                "2c": ["2c", "2C", "two of clubs"], "2d": ["2d", "2D", "two of diamonds"],
                "2h": ["2h", "2H", "two of hearts"], "2s": ["2s", "2S", "two of spades"],
                "3c": ["3c", "3C", "three of clubs"], "3d": ["3d", "3D", "three of diamonds"],
                "3h": ["3h", "3H", "three of hearts"], "3s": ["3s", "3S", "three of spades"],
                "4c": ["4c", "4C", "four of clubs"], "4d": ["4d", "4D", "four of diamonds"],
                "4h": ["4h", "4H", "four of hearts"], "4s": ["4s", "4S", "four of spades"],
                "5c": ["5c", "5C", "five of clubs"], "5d": ["5d", "5D", "five of diamonds"],
                "5h": ["5h", "5H", "five of hearts"], "5s": ["5s", "5S", "five of spa   des"],
                "6c": ["6c", "6C", "six of clubs"], "6d": ["6d", "6D", "six of diamonds"],
                "6h": ["6h", "6H", "six of hearts"], "6s": ["6s", "6S", "six of spades"],
                "7c": ["7c", "7C", "seven of clubs"], "7d": ["7d", "7D", "seven of diamonds"],
                "7h": ["7h", "7H", "seven of hearts"], "7s": ["7s", "7S", "seven of spades"],
                "8c": ["8c", "8C", "eight of clubs", "eigth of clubs"], "8d": ["8d", "8D", "eight of diamonds"],
                "8h": ["8h", "8H", "eight of hearts"], "8s": ["8s", "8S", "eight of spades"],
                "9c": ["9c", "9C", "nine of clubs"], "9d": ["9d", "9D", "nine of diamonds"],
                "9h": ["9h", "9H", "nine of hearts"], "9s": ["9s", "9S", "nine of spades"],
                "10c": ["10c", "10C", "ten of clubs"], "10d": ["10d", "10D", "ten of diamonds"],
                "10h": ["10h", "10H", "ten of hearts"], "10s": ["10s", "10S", "ten of spades"],
                "Jc": ["Jc", "JC", "jack of clubs"], "Jd": ["Jd", "JD", "jack of diamonds"],
                "Jh": ["Jh", "JH", "jack of hearts"], "Js": ["Js", "JS", "jack of spades"],
                "Qc": ["Qc", "QC", "queen of clubs"], "Qd": ["Qd", "QD", "queen of diamonds"],
                "Qh": ["Qh", "QH", "queen of hearts"], "Qs": ["Qs", "QS", "queen of spades"],
                "Kc": ["Kc", "KC", "king of clubs"], "Kd": ["Kd", "KD", "king of diamonds"],
                "Kh": ["Kh", "KH", "king of hearts"], "Ks": ["Ks", "KS", "king of spades"],
                "JOKER_black": ["JOKER_black", "JOKER"], "JOKER_red": ["JOKER_red"]
               } # set of all class names found throughout the datasets -> change them all to my chosen namings

def parse_xml(path, img_dims: bool=False):
    tree = ET.parse(path)
    symbols = []
    for obj in tree.findall('object'):
        name = obj.find('name').text

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        if not img_dims:
            symbols.append([name, xmin, ymin, xmax, ymax])
        else:
            img_h = int(tree.find("size").find("height").text)
            img_w = int(tree.find("size").find("width").text)
            symbols.append([name, xmin, ymin, xmax, ymax, img_h, img_w])

    return symbols

# loads the xmls and writes them to a corresponding text file, they aren't processed at all
# processing will be handled by a separate function
def load_xmls(path, name):
    xml_files = glob.glob(os.path.join(path, '*.xml'))
    print(os.path.join(path, '*.xml'))
    print(os.path.isdir(path), path)
    print(xml_files)
    content = []
    for xml_file in xml_files:
        file_name = xml_file.split(os.sep)[-1][:-4]
        print(file_name)
        symbols = parse_xml(xml_file)
        for symbol in symbols:
            cls, xmin, ymin, xmax, ymax = symbol
            print(file_name, cls, xmin, ymin, xmax, ymax)
            content.append([file_name, cls, xmin, ymin, xmax, ymax])
    output_name = path + "\\" + name + "_annotations.txt"
    np.savetxt(output_name, np.array(content), fmt='%s')

# flag for datasets where we should ignore non-digits, due to unstandard representations for letters
# separate_img_path refers to the dataset that has its images and annotations stored in separate folders
def write_labels(file_path, ignore_non_digits=False, separate_img_path=None):
    global all_classes

    final_path = os.path.join(Path(os.getcwd()).parents[1], "data\\")
    labels_fname = "train_labels_RAW.txt"

    STD_COMPONENTS = 6
    # should restrict the amount of cards we take from each dataset, since the third dataset is gigantic and could
    # pull our whole dataset with it
    final_results = dict()
    file_dir = os.path.dirname(file_path)
    print(file_dir)
    with open(file_path, 'r') as f:
        with open(final_path + labels_fname, "a") as fw:
            lines = f.readlines()
            for line in lines:
                contents = line.split()
                img_name, *class_names, xmin, ymin, xmax, ymax = contents
                final_cls = ""
                if len(contents) > STD_COMPONENTS:
                    # the third dataset has annoyingly long class names
                    cls = " ".join(contents[1:4])
                else:
                    cls = contents[1]
                for classes in all_classes:
                    if cls in all_classes[classes]:
                        final_cls = classes
                        break
                if ignore_non_digits and final_cls in ["As", "Ad", "Ac", "Ah", "Js", "Jd", "Jc", "Jh", "Qs", "Qd", "Qc", "Qh",
                                                       "Ks", "Kd", "Kc", "Kh"]:
                    continue
                if cls not in final_results:
                    final_results[cls] = 1
                else:
                    final_results[cls] += 1
                if final_results[cls] >= 55:
                    continue
                if separate_img_path is None:
                    fw.write(file_dir + "\\" + img_name + ".jpg" + "," + final_cls + "," + xmin + "," + ymin + "," + xmax + "," + ymax + "\n")
                else:
                    fw.write(separate_img_path + "\\" + img_name + ".jpg" + "," + final_cls + "," + xmin + "," + ymin + "," + xmax + "," + ymax + "\n")

"""
Path is absolute path to xml files root.
"""
def clean_xmls(path):
    global all_classes

    xml_files = glob.glob(os.path.join(path, "*.xml"))
    for xml_file in tqdm(xml_files):
        xml_tree = ET.parse(xml_file)
        for obj in xml_tree.findall('object'):
            name = obj.find('name')

            for cls in all_classes:
                if name.text in all_classes[cls]:
                    name.text = cls
            xml_tree.write(xml_file)

# due to EXIF metadata, images will appear rotated in the labelimg program
# BUT are not actually rotated!!! I had to use a special software that automatically rotates images captured with a phone
# camera, however this lead to my labels being all wrong. fortunately, using this function I managed to rotate all wrong
# labels to the new, correct format. outside of this case, this function is entirely useless
def reverse_xml(path):
    xml_files = glob.glob(os.path.join(path, '*.xml'))
    image_files = glob.glob(os.path.join(path, '*.jpg'))
    print(len(xml_files))
    for file_index, xml_file in tqdm(enumerate(xml_files)):
        file_name = xml_file.split(os.sep)[-1][:-4]
        # print(image_files[file_index])
        img = cv.imread(image_files[file_index])
        # print(img.shape)
        tree = ET.parse(xml_file)
        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin')
            ymin = bbox.find('ymin')
            xmax = bbox.find('xmax')
            ymax = bbox.find('ymax')
            # xmin.text, ymin.text = ymin.text, xmin.text
            # xmax.text, ymax.text = ymax.text, xmax.text
            # xmin_t, xmax_t = int(xmin.text), int(xmax.text)
            # xmin.text = str(img.shape[1] - xmin_t)
            # # xmax.text = str( - xmax_t + img.shape[1])
            # xmin.text = str(max(min(int(xmin.text), img.shape[1]), 0))
            # ymin.text = str(max(min(int(ymin.text), img.shape[0]), 0))
            # xmax.text = str(max(min(int(xmax.text), img.shape[1]), 0))
            # ymax.text = str(max(min(int(ymax.text), img.shape[0]), 0))
            # if int(xmin.text) > int(xmax.text):
            #     xmin.text, xmax.text = xmax.text, xmin.text
        tree.write(xml_file)

# simply run some statistics to get an idea of current, un-augmented dataset
# for example, average examples per class
def dataset_statistics_unaugmented():
    final_path = os.path.join(Path(os.getcwd()).parents[1], "data\\")
    labels_fname = "train_labels_RAW.txt"

    with open(final_path + labels_fname, "r") as f:
        lines = f.readlines()
        final_results = dict()
        for line in lines:
            img_name, cls, *_ = line.split(",")
            if cls not in final_results:
                final_results[cls] = 1
            else:
                final_results[cls] += 1
        sum = 0
        for cls in final_results:
            sum += final_results[cls]
        mean = sum/(len(final_results))
        sum = 0
        for cls in final_results:
            sum += (final_results[cls] - mean) ** 2
        variance = sum / (len(final_results))
        print(f"Average examples per class is {mean}")
        print(f"Variance is {variance}")
        for cls in final_results:
            print(f"Class {cls} has {final_results[cls]} examples.")

def dataset_statistics_augmented():
    final_path = os.path.join(Path(os.getcwd()).parents[1], "data\\")
    labels_fname = "train_labels_RAW.txt"

    with open(final_path + labels_fname, "r") as f:
        lines = f.readlines()
        final_results = dict()
        for line in lines:
            img_name, cls, *_ = line.split(",")
            if cls not in final_results:
                final_results[cls] = 20
            else:
                final_results[cls] += 20
        sum = 0
        for cls in final_results:
            sum += final_results[cls]
        mean = sum/(len(final_results))
        sum = 0
        for cls in final_results:
            sum += (final_results[cls] - mean) ** 2
        variance = sum / (len(final_results))
        print(f"Average examples per class is {mean}")
        print(f"Variance is {variance}")
        for cls in final_results:
            print(f"Class {cls} has {final_results[cls]} examples.")

def write_unaugmented():
    with open(os.path.join(Path(os.getcwd()).parents[1], "data\\train_labels_RAW.txt"), "r") as f:
        lines = f.readlines()
        for lindex, line in enumerate(lines):
            path, cls, *coords = line.split(",")
            xmin, ymin, xmax, ymax = map(int, coords)
            img_source = cv.imread(path)
            print(path)
            print(img_source.shape, xmin, ymin, xmax, ymax)
            cv.imwrite(os.path.join(Path(os.getcwd()).parents[1], "data\\train_imgs\\" + str(lindex) + ".jpg"),
                       img_source[ymin:ymax, xmin:xmax])

def write_full_unaugmented():
    with open(os.path.join(Path(os.getcwd()).parents[1], "data\\train_labels_RAW.txt"), "r") as f:
        lines = f.readlines()
        def cmp_lines(l1, l2):
            if l1 < l2:
                return -1
            elif l2 > l1:
                return 1
            return 0
        lines = sorted(lines, key=cmp_to_key(cmp_lines))
        with open(os.path.join(Path(os.getcwd()).parents[1], "data\\train_labels_full.txt"), "w") as fw:
            fw.writelines(lines)

if __name__ == "__main__":
    DATASET1 = "data\\RAW\\my-stuff\\"
    DATASET1_LAB_NAMES = "my-stuff"
    DATASET2 = "data\\RAW\\Kaghle-playing-cards-images-object-detection-dataset\\train\\train\\"
    DATASET2_LAB_NAMES = "Kaghle-playing-cards-images-object-detection-dat"
    DATASET3 = "data\\RAW\\Kaggle-the-complete-playing-card-dataset\\Annotations\\Annotations\\"
    DATASET3_LAB_NAMES = "Kaggle-the-complete-playing-card-dataset"
    DATASET4 = "data\\RAW\\Kaggle-playing-cards-labelized-dataset\\train_zipped\\"
    DATASET4_LAB_NAMES = "Kaggle-playing-cards-labelized-dataset"

    # load_xmls(os.path.join(Path(os.getcwd()).parents[1], DATASET1),
    #           DATASET1_LAB_NAMES)
    # load_xmls(os.path.join(Path(os.getcwd()).parents[1], DATASET2),
    #           DATASET2_LAB_NAMES)
    # load_xmls(os.path.join(Path(os.getcwd()).parents[1], DATASET3),
    #           DATASET3_LAB_NAMES)
    # # the last one has bizzare kings, queens, aces and jacks, should ignore them
    # load_xmls(os.path.join(Path(os.getcwd()).parents[1], DATASET4),
    #           DATASET4_LAB_NAMES)
    #
    # write_labels(os.path.join(Path(os.getcwd()).parents[1], DATASET1 + DATASET1_LAB_NAMES + "_annotations.txt"))
    # write_labels(os.path.join(Path(os.getcwd()).parents[1], DATASET2 + DATASET2_LAB_NAMES + "_annotations.txt"))
    # write_labels(os.path.join(Path(os.getcwd()).parents[1], DATASET3 + DATASET3_LAB_NAMES + "_annotations.txt"),
    #              separate_img_path="D:\\facultate stuff\\licenta\\data\\RAW\\Kaggle-the-complete-playing-card-dataset\\Images\\Images")
    # write_labels(os.path.join(Path(os.getcwd()).parents[1], DATASET4 + DATASET4_LAB_NAMES + "_annotations.txt"), True)
    dataset_statistics_unaugmented()
    # dataset_statistics_augmented()
    # write_full_unaugmented()
    # reverse_xml(os.path.join(Path(os.getcwd()).parents[1], DATASET1))
    # write_unaugmented()
    # clean_xmls("D:\\facultate stuff\\licenta\\data\\train_imgs_full\\")

