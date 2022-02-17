import os
import glob
import pdb
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

possible_classes = {"Ac", "Ad", "Ah", "As",
                    "1c", "1d", "1h", "1s",
                    "2c", "2d", "2h", "2s",
                    "3c", "3d", "3h", "3s",
                    "4c", "4d", "4h", "4s",
                    "5c", "5d", "5h", "5s",
                    "6c", "6d", "6h", "6s",
                    "7c", "7d", "7h", "7s",
                    "8c", "8d", "8h", "8s",
                    "9c", "9d", "9h", "9s",
                    "10c", "10d", "10h", "10s",
                    "Jc", "Jd", "Jh", "Js",
                    "Qc", "Qd", "Qh", "Qs",
                    "Kc", "Kd", "Kh", "Ks",
                    "JOKER_red", "JOKER_black"
                    "not a card"
                    } # set of all possible classes


def parse_xml(path):
    tree = ET.parse(path)
    symbols = []
    for obj in tree.findall('object'):
        name = obj.find('name').text

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        symbols.append([name, xmin, ymin, xmax, ymax])

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
        # nume_fisier = xml_file.split(os.sep)[-2]
        file_name = xml_file.split(os.sep)[-1][:-4]
        # nume_fisier = xml_file
        print(file_name)
        symbols = parse_xml(xml_file)
        for symbol in symbols:
            name = symbol[0]
            xmin = symbol[1]
            ymin = symbol[2]
            xmax = symbol[3]
            ymax = symbol[4]
            print(file_name, name, xmin, ymin, xmax, ymax)
            content.append([file_name, name, xmin, ymin, xmax, ymax])
    output_name = path + "\\" + name + "_annotations.txt"
    np.savetxt(output_name, np.array(content), fmt='%s')

if __name__ == "__main__":
    load_xmls(os.path.join(Path(os.getcwd()).parents[1], "data\\RAW\\my-stuff\\"),
              "my-stuff")
    load_xmls(os.path.join(Path(os.getcwd()).parents[1], "data\\RAW\\Kaghle-playing-cards-images-object-detection-dataset\\train\\train\\"),
              "Kaghle-playing-cards-images-object-detection-dat")
    load_xmls(os.path.join(Path(os.getcwd()).parents[1], "data\\RAW\\Kaggle-the-complete-playing-card-dataset\\Annotations\\Annotations"),
              "Kaggle-the-complete-playing-card-dataset")
    # the last one has bizzare kings, queens and valleys, should ignore them
    load_xmls(os.path.join(Path(os.getcwd()).parents[1], "data\\RAW\\Kaggle-playing-cards-labelized-dataset\\train_zipped"),
              "Kaggle-playing-cards-labelized-dataset")