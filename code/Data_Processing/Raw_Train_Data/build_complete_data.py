from tqdm import tqdm
from PIL import Image

from global_constants import CARD_WIDTH, CARD_HEIGHT
from augment_data import MULTIPLIER

if __name__ == "__main__":
    unaug_labels_path = "D:\\facultate stuff\\licenta\\data\\train_labels_RAW.txt"
    unaug_imgs_path = "D:\\facultate stuff\\licenta\\data\\train_imgs\\"
    aug_imgs_path = "D:\\facultate stuff\\licenta\\data\\train_imgs_aug\\"
    complete_labels = "D:\\facultate stuff\\licenta\\data\\train.txt"
    complete_img_path = "D:\\facultate stuff\\licenta\\data\\train\\"

    with open(unaug_labels_path, "r") as f:
        with open(complete_labels, "w+") as fw:
            lines = f.readlines()
            for lindex, line in tqdm(enumerate(lines)):
                img_path, cls, *coords = line.split(",")
                with Image.open(unaug_imgs_path + str(lindex) + ".jpg") as img:
                    img = img.resize((CARD_WIDTH, CARD_HEIGHT))
                    img.save(complete_img_path + str(lindex * (MULTIPLIER + 1)) + '.jpg')
                fw.write(complete_img_path + str(lindex * (MULTIPLIER + 1)) + '.jpg,' + cls + '\n')
                for m in range(MULTIPLIER):
                    with Image.open(aug_imgs_path + str(m * len(lines) + lindex) + ".jpeg") as img:
                        img.save(complete_img_path + str((m + 1) + lindex * (MULTIPLIER + 1)) + ".jpg")
                    fw.write(complete_img_path + str((m + 1) + lindex * (MULTIPLIER + 1)) + '.jpg,' + cls + '\n')
