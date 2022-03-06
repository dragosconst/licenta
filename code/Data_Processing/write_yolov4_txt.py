import glob
import os
from tqdm import tqdm

import xml.etree.ElementTree as ET
from Data_Processing.Raw_Train_Data.raw import possible_classes

def write_data():
    xml_files = glob.glob(os.path.join("D:\\facultate stuff\\licenta\\data\\train_imgs_full\\", "*.xml"))
    with open("D:\\facultate stuff\\licenta\\data\\train_imgs_full\\train.txt", "w+") as fw:
        for xml_file in tqdm(xml_files):
            xml_tree = ET.parse(xml_file)
            fw.write(xml_file[:-3] + "jpg ")
            for obj in xml_tree.findall('object'):
                name = obj.find('name').text

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                fw.write(f"{xmin},{ymin},{xmax},{ymax},{possible_classes[name]} ")
            fw.write("\n")

if __name__ == "__main__":
    write_data()