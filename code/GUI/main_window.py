from Windows_utils.windows import grab_all_open_windows, grab_selected_window_contents
import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
import threading
import time

# window names
MAIN_WINDOW = "Main"
CAM_W = "cameraw"
DEBUG_STATE = True
MW_W = 1000
MW_H = 630
FS = 60
CR_PROCESS = "cr_proc"
CAM_W_IMG = "cam_img"
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

# def get_selected_window_texture(win_name):
#     if not dpg.does_item_exist(CR_PROC_TEXT):
#         return
#     print("aaaaa")
#     global img_normalized
#     w = 800
#     h = 600
#     img = grab_selected_window_contents(win_name, w, h)
#     # cv.imshow("img", img)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
#     img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
#     img = img.flatten().astype(np.float32)
#     img_normalized = img / 255
#     dpg.set_value(CR_PROC_TEXT, img_normalized)
#
# def _change_active_window(sender, app_data, user_data):
#     global CR_PROCESS, img_normalized
#     if not dpg.does_item_exist(CR_PROCESS):
#         with dpg.window(tag=CR_PROCESS):
#             w = 800
#             h = 600
#             img = grab_selected_window_contents(app_data, w, h)
#             imc = img.copy()
#             img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
#             img = img.flatten().astype(np.float32)
#             img_normalized = img / 255
#             with dpg.texture_registry(show=False):
#                 dpg.add_raw_texture(w, h, img_normalized, tag=CR_PROC_TEXT)
#             dpg.add_image(CR_PROC_TEXT)
#         dpg.show_item(CR_PROCESS)
#         print("finished change")

def update_camera():
    if not dpg.does_item_exist(CAM_W_IMG):
        return
    ret, frame = video_capture.read()
    if frame is None:
        return
    img_data = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_data = img_data.flatten().astype(np.float32)
    img_normalized = img_data / 255
    dpg.set_value(CAM_W_IMG, img_normalized)

def _release_video_capture(sender, app_data, user_data):
    print("okay")
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

        with dpg.window(tag=CAM_W, on_close=_release_video_capture):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(frame_width, frame_height, img_normalized, tag=CAM_W_IMG, format=dpg.mvFormat_Float_rgb)
            dpg.add_image(CAM_W_IMG)
    else:
        video_capture.open(0)
        dpg.show_item(CAM_W)



def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        win_names = grab_all_open_windows()
        print(win_names)
        dpg.add_button(label="Capture camera", width=MW_W, height=MW_H // 4, callback=_capture_camera)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Select game", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4, tag="Debug", callback=_change_debug_text, user_data="Debug")
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

