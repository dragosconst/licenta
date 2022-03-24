import time

from torch.optim import Adam, SGD
import torch.utils.data
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

from GUI.main_window import create_main_window, create_dpg_env, update_camera, update_screen_area, get_selected_window_texture
import dearpygui.dearpygui as dpg
from Models.Vision.faster import train_frcnn, validate, train_fccnn_reference, get_faster
from Utils.utils import load_dataloader, get_loader

def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    frcnn = get_faster("../data/frcnn_resnet50_5k_per_class.pt")
    frcnn.eval()
    while dpg.is_dearpygui_running():
        update_camera(frcnn)
        update_screen_area(frcnn)
        get_selected_window_texture()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()