from Windows_utils.windows import grab_all_open_windows, grab_selected_window_contents, grab_screen_area
from Image_Processing.line_detection import get_lines_in_image
import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
import threading
import time

# window names and other constants
MAIN_WINDOW = "Main"
CAM_W = "cameraw"
CAM_C = "cameracanny"
DEBUG_STATE = True
MW_W = 1000
MW_H = 630
FS = 60
CR_PROCESS = "cr_proc"
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
GAMES = ["Blackjack"]
img_normalized = None
video_capture = None

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

def update_camera():
    if not dpg.does_item_exist(CAM_W_IMG):
        return
    ret, frame = video_capture.read()
    if frame is None:
        return
    img_data = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_data, canny = get_lines_in_image(img_data)
    img_data = img_data.flatten().astype(np.float32)
    canny = canny.flatten().astype(np.float32)
    img_normalized = img_data / 255
    canny = canny / 255
    dpg.set_value(CAM_W_IMG, img_normalized)
    dpg.set_value(CAM_C_IMG, canny)

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
                dpg.add_ratexture(frame_width, frame_height, img_normalized, tag=CAM_W_IMG, format=dpg.mvFormat_Float_rgb)
            dpg.add_image(CAM_W_IMG)
        with dpg.window(tag=CAM_C, label="Camera"):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(frame_width, frame_height, img_normalized, tag=CAM_C_IMG, format=dpg.mvFormat_Float_rgb)
            dpg.add_image(CAM_C_IMG)
    else:
        video_capture.open(0)
        dpg.show_item(CAM_W)

def update_screen_area():
    if not dpg.does_item_exist(DIM_W) or not dpg.is_item_visible(DIM_W):
        return
    w = dpg.get_value(SC_WH) if dpg.get_value(SC_WH) > 100 else 100
    h = dpg.get_value(SC_HH) if dpg.get_value(SC_HH) > 100 else 100
    x = dpg.get_value(SC_X)
    y = dpg.get_value(SC_Y)
    img = grab_screen_area(x, y, w, h)
    img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
    img = cv.resize(img, (SC_STDW, SC_STDH))
    img = img.flatten().astype(np.float32)
    img_normalized = img / 255
    dpg.set_value(SC_C, img_normalized)

def _get_screen_area():
    if not dpg.does_item_exist(DIM_W):
        with dpg.window(tag=DIM_W, label="Dimensions"):
            dpg.add_slider_int(label="Width", tag=SC_WH, default_value=800, min_value=100, max_value=1000)
            dpg.add_slider_int(label="Height", tag=SC_HH, default_value=600, min_value=100, max_value=1000)
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

def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        dpg.add_button(label="Capture camera", width=MW_W, height=MW_H // 4, callback=_capture_camera)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4, callback=_get_screen_area)
        dpg.add_combo(label="Select game", items=GAMES)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4, tag="Debug", callback=_change_debug_text, user_data="Debug")
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

