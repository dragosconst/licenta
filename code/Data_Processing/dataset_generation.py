from PIL import Image
import numpy as np

from Utils.file_utils import get_image_files, get_random_bg_img
"""
Generate a dataset of a certain size from a given dataset of cropped cards and another dataset of random backgrounds.

"""

IM_HEIGHT = 1080
IM_WIDTH = 1900
def generate_random_image(*cards, bg_image: Image):
    """

    :param cards: tuples of (img, data), where img is the pic of the card and data represents the relevant labeling data
    :param bg_image: a random background image
    :return:
    """
    assert cards is not None

    card_masks = []
    for (img, data) in cards:
        img_full = np.zeros((IM_WIDTH, IM_HEIGHT))