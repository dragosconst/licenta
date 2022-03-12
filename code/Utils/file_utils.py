from typing import List, Optional, Dict, Tuple
import glob
import os

import numpy as np
from PIL import Image

def get_image_files() -> List[str]:
    dir = "D:\\facultate stuff\\licenta\\data\\RAW\\dtd\\images"
    imgs_fns = []
    for subdirs in glob.glob(dir + "\\*"):
        for img_fn in glob.glob(os.path.join(subdirs, "*.jpg")):
            imgs_fns.append(img_fn)
    return imgs_fns

def get_random_bg_img(imgs_fns) -> Image:
    idx = np.random.randint(0, len(imgs_fns))
    img = Image.open(imgs_fns[idx])
    return img
