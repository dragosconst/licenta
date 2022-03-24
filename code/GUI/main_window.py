import threading
import time

import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from Windows_utils.windows import grab_all_open_windows, grab_selected_window_contents, grab_screen_area
from Image_Processing.line_detection import get_lines_in_image
from Image_Processing.detection_draw import draw_detection
from Data_Processing.detection_processing_resnet import second_nms, filter_under_thresh

# window names and other constants
MAIN_WINDOW = "Main"
CAM_W = "cameraw"
CAM_C = "cameracanny"
DEBUG_STATE = True
MW_W = 1000
MW_H = 630
FS = 60
CR_PROCESS = "cr_proc"
CR_PROC_TEXT = "cr_proc_text"
CAM_W_IMG = "cam_img"
CAM_C_IMG = "cam_c_img"
DIM_W = "dimw"
SC_WH = "dimwh"
SC_HH = "dimhh"
SC_STDW = 800
SC_STDH = 600
SC_X = "posx"
SC_Y = "posy"
SC_W = "screencw"
SC_C = "screencap"
SEL_W = "selectw"
GAMES = ["Blackjack"]
img_normalized = None
video_capture = None
UPDATE_CAMERA_RATE = 1000
last_camera_update = 0
dets = []
selected_window = None

def create_dpg_env():
    dpg.create_context()
    with dpg.font_registry():
        default_font =  dpg.add_font('../data/fonts/NotoSerifCJKjp-Medium.otf', FS)
    dpg.create_viewport(title="Smart Cards", width=MW_W, height=MW_H)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    return default_font

def _change_debug_text(sender, app_data, user_data):
    global DEBUG_STATE
    DEBUG_STATE = not DEBUG_STATE
    dpg.configure_item(user_data, label="Debug mode:" + ("ON" if DEBUG_STATE else "OFF"))


@torch.inference_mode()
def update_camera(model: torch.nn.Module):
    global last_camera_update, UPDATE_CAMERA_RATE, dets

    if not dpg.does_item_exist(CAM_W_IMG):
        return
    ret, frame = video_capture.read()
    if frame is None:
        return
    img_data = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_data)
    img_tensor = img_tensor.permute(2, 0, 1)
    shape = img_tensor.size()[1:]
    img_tensor = torch.from_numpy(np.asarray(T.ToPILImage()(img_tensor).resize((1900, 1080))))
    img_tensor = img_tensor.permute(2, 0, 1)
    if time.time() * 1000 - last_camera_update > UPDATE_CAMERA_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        detections = model(img_tensor.unsqueeze(0))
        dets = detections[0]

    img_pil = draw_detection(T.ToPILImage()(img_tensor), dets)
    img_pil = img_pil.resize(shape[::-1])
    img_data = np.asarray(img_pil)
    img_data = img_data.flatten().astype(np.float32)
    img_normalized = img_data / 255
    dpg.set_value(CAM_W_IMG, img_normalized)

def _release_video_capture(sender, app_data, user_data):
    video_capture.release()

def _capture_camera(sender, app_data, user_data):
    global video_capture, img_normalized
    if not dpg.does_item_exist(CAM_W):
        video_capture = cv.VideoCapture(0)
        ret, frame = video_capture.read()
        img_data = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_data = img_data.flatten().astype(np.float32)
        img_normalized = img_data / 255
        frame_width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
        frame_height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        with dpg.window(tag=CAM_W, on_close=_release_video_capture, label="Camera"):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(frame_width, frame_height, img_normalized, tag=CAM_W_IMG, format=dpg.mvFormat_Float_rgb)
            dpg.add_image(CAM_W_IMG)
    else:
        video_capture.open(0)
        dpg.show_item(CAM_W)

@torch.inference_mode()
def update_screen_area(model: torch.nn.Module):
    global last_camera_update, UPDATE_CAMERA_RATE, dets

    if not dpg.does_item_exist(DIM_W) or not dpg.is_item_visible(DIM_W):
        return
    w = dpg.get_value(SC_WH) if dpg.get_value(SC_WH) > 100 else 100
    h = dpg.get_value(SC_HH) if dpg.get_value(SC_HH) > 100 else 100
    x = dpg.get_value(SC_X)
    y = dpg.get_value(SC_Y)
    img = grab_screen_area(x, y, w, h)

    img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
    img_tensor = torch.from_numpy(img[:, :, :3])
    img_tensor = img_tensor.permute(2, 0, 1)
    shape = img_tensor.size()[1:]
    img_tensor = torch.from_numpy(np.asarray(T.ToPILImage()(img_tensor).resize((1900, 1080))))
    img_tensor = img_tensor.permute(2, 0, 1)
    if time.time() * 1000 - last_camera_update > UPDATE_CAMERA_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        detections = model(img_tensor.unsqueeze(0))
        dets = detections[0]
        filter_under_thresh(dets)
        second_nms(dets)

    img_pil = draw_detection(T.ToPILImage()(img_tensor), dets)
    img_pil = img_pil.resize(shape[::-1])
    img[:, :, :3] = np.asarray(img_pil)
    img = cv.resize(img, (SC_STDW, SC_STDH))
    img = img.flatten().astype(np.float32)
    img_normalized = img / 255
    dpg.set_value(SC_C, img_normalized)

def _get_screen_area():
    if not dpg.does_item_exist(DIM_W):
        with dpg.window(tag=DIM_W, label="Dimensions"):
            dpg.add_slider_int(label="Width", tag=SC_WH, default_value=1900, min_value=100, max_value=2000)
            dpg.add_slider_int(label="Height", tag=SC_HH, default_value=1080, min_value=100, max_value=2000)
            dpg.add_slider_int(label="X pos", tag=SC_X, min_value=0, max_value=1900)
            dpg.add_slider_int(label="Y pos", tag=SC_Y, min_value=0, max_value=1200)
        with dpg.window(tag=SC_W, label="Screen Capture"):
            with dpg.texture_registry(show=False):
                w = dpg.get_value(SC_WH)
                h = dpg.get_value(SC_HH)
                x = dpg.get_value(SC_X)
                y = dpg.get_value(SC_Y)
                img = grab_screen_area(x, y, w, h)
                img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
                img = cv.resize(img, (SC_STDW, SC_STDH))
                img = img.flatten().astype(np.float32)
                img_normalized = img / 255
                dpg.add_raw_texture(SC_STDW, SC_STDH, img_normalized, tag=SC_C, format=dpg.mvFormat_Float_rgba)
            dpg.add_image(SC_C)
    else:
        dpg.show_item(DIM_W)
        dpg.show_item(SC_W)

@torch.inference_mode()
def get_selected_window_texture(model: torch.nn.Module):
    global selected_window, img_normalized, last_camera_update, UPDATE_CAMERA_RATE, dets

    if not dpg.does_item_exist(CR_PROC_TEXT):
        return
    w = 1900
    h = 1080
    img = grab_selected_window_contents(selected_window, w, h)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
    img_tensor = torch.from_numpy(img[:, :, :3])
    img_tensor = img_tensor.permute(2, 0, 1)
    shape = img_tensor.size()[1:]
    # img_tensor = torch.from_numpy(np.asarray(T.ToPILImage()(img_tensor).resize((1900, 1080))))
    # img_tensor = img_tensor.permute(2, 0, 1)
    if time.time() * 1000 - last_camera_update > UPDATE_CAMERA_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        detections = model(img_tensor.unsqueeze(0))
        dets = detections[0]
        filter_under_thresh(dets)
        second_nms(dets)

    img_pil = draw_detection(T.ToPILImage()(img_tensor), dets)
    # img_pil = img_pil.resize(shape[::-1])
    img[:, :, :3] = np.asarray(img_pil)
    img = img.flatten().astype(np.float32)
    img_normalized = img / 255
    dpg.set_value(CR_PROC_TEXT, img_normalized)

def _change_active_window(sender, app_data, user_data):
    global CR_PROCESS, img_normalized, selected_window

    if not dpg.does_item_exist(CR_PROCESS):
        with dpg.window(tag=CR_PROCESS):
            w = 1900
            h = 1080
            selected_window = app_data
            img = grab_selected_window_contents(app_data, w, h)
            imc = img.copy()
            # img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
            cv.imshow("pppp", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            img = img.flatten().astype(np.float32)
            img_normalized = img / 255
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(w, h, img_normalized, tag=CR_PROC_TEXT)
            dpg.add_image(CR_PROC_TEXT)
        dpg.show_item(CR_PROCESS)

def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        dpg.add_button(label="Capture camera", width=MW_W, height=MW_H // 4, callback=_capture_camera)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4, callback=_get_screen_area)
        win_names = grab_all_open_windows()
        print(win_names)
        dpg.add_combo(label="Select window", tag=SEL_W, items=win_names, width=MW_W // 2,
                      callback=_change_active_window)

        dpg.add_combo(label="Select game", items=GAMES)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4, tag="Debug", callback=_change_debug_text, user_data="Debug")
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

