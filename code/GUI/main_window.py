import threading
import time
from typing import Tuple, List, Dict
import dill as pickle

import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from Windows_utils.windows import grab_all_open_windows, grab_selected_window_contents, grab_screen_area
from Image_Processing.line_detection import get_lines_in_image
from Data_Processing.Raw_Train_Data.raw import pos_cls_inverse
from Image_Processing.detection_draw import draw_detection
from Data_Processing.detection_processing_resnet import second_nms, filter_under_thresh, filter_detections_by_game, filter_non_group_detections
from Data_Processing.group_filtering import get_player_hand
from Data_Processing.extra_filters import filter_same_card
from Data_Processing.frame_hand_detection import compare_detections
from Game_Engines.bj_enginge import BlackjackEngine
from Game_Engines.base_engine import BaseEngine

# window names and other constants

# windows (not the os) specific constants
# ---------------------------------------
MAIN_WINDOW = "Main"
CAM_W = "cameraw"
CAM_C = "cameracanny"
DEBUG_STATE = True
MW_W = 1000
MW_H = 630
FS = 60
CR_PROCESS = "cr_proc"
CR_PROC_DIM = "cr_proc_dim"
CR_PROC_TEXT = "cr_proc_text"
CR_PROC_WH = "cr_proc_dimwh"
CR_PROC_HH = "cr_proc_dimhh"
CR_PROC_X = "cr_proc_posx"
CR_PROC_Y = "cr_proc_posy"
CR_PROC_AG_BUT = "cr_proc_ag_but"
CAM_W_IMG = "cam_img"
CAM_C_IMG = "cam_c_img"
DIM_W = "dimw"
SC_WH = "dimwh"
SC_HH = "dimhh"
SC_STDW = 800
SC_STDH = 600
FRCNN_W = 1900
FRCNN_H = 1080
SC_X = "posx"
SC_Y = "posy"
SC_W = "screencw"
SC_C = "screencap"
SEL_W = "selectw"

# cards game specific constants
# -----------------------------
GAMES = ["Blackjack", "Razboi", "Macao", "Septica"]
current_game = None # type: str
current_game_engine = None # type: BaseEngine

# image and net specific constants
# --------------------------------
img_normalized = None
video_capture = None
UPDATE_DETECTIONS_RATE = 400 # ms interval at which to update our detections, used to avoid slowing down the program too much
last_camera_update = 0
dets = {} # type: Dict[str, torch.Tensor]
selected_window_title = None
selected_window_hwnd = None
same_ph_in_a_row = 0
same_cp_in_a_row = 0
last_player_hand = []
last_card_pot = []
DET_ROW_THRES = 2
cards_pot = []
cards_pot_labels = []
player_hand = []
player_hand_labels = []


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
    global last_camera_update, UPDATE_DETECTIONS_RATE, dets, video_capture

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
    if time.time() * 1000 - last_camera_update > UPDATE_DETECTIONS_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        detections = model(img_tensor.unsqueeze(0))
        dets = detections[0]
        apply_inplace_filters(dets)

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
    global last_camera_update, UPDATE_DETECTIONS_RATE, dets, cards_pot, player_hand, img_normalized

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
    if time.time() * 1000 - last_camera_update > UPDATE_DETECTIONS_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        detections = model(img_tensor.unsqueeze(0))
        dets = detections[0]
        apply_inplace_filters(dets)
        cards_pot, player_hand = get_player_hand(current_game, dets)
        cards_pot, player_hand = filter_same_card(dets, cards_pot, player_hand)
        check_if_change_detections(dets)

    img_pil = draw_detection(T.ToPILImage()(img_tensor), dets, cards_pot, player_hand)
    img_pil = img_pil.resize((min(dpg.get_viewport_width(), FRCNN_W), min(dpg.get_viewport_height(), FRCNN_H)))
    # img_pil = img_pil.resize(shape[::-1])
    img[:dpg.get_viewport_height(), :dpg.get_viewport_width(), :3] = np.asarray(img_pil)
    # img_pil = img_pil.resize(shape[::-1])
    # img[:, :, :3] = np.asarray(img_pil)
    # img = cv.resize(img, (SC_STDW, SC_STDH))
    img = img.flatten().astype(np.float32)
    img_normalized = img / 255
    dpg.set_value(SC_C, img_normalized)


def _get_screen_area():
    global img_normalized

    if not dpg.does_item_exist(DIM_W):
        with dpg.window(tag=SC_W, width=FRCNN_W, height=FRCNN_H, label="Screen Capture"):
            dpg.add_button(label="Restart Agent", tag=CR_PROC_AG_BUT, callback=_restart_agent)
            with dpg.texture_registry(show=False):
                w = 1900
                h = 1080
                x = 0
                y = 0
                img = grab_screen_area(x, y, w, h)
                img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
                # img = cv.resize(img, (SC_STDW, SC_STDH))
                img = img.flatten().astype(np.float32)
                img_normalized = img / 255
                dpg.add_raw_texture(w, h, img_normalized, tag=SC_C, format=dpg.mvFormat_Float_rgba)
            with dpg.window(tag=DIM_W, label="Dimensions"):
                dpg.add_slider_int(label="Width", tag=SC_WH, default_value=1900, min_value=100, max_value=2000)
                dpg.add_slider_int(label="Height", tag=SC_HH, default_value=1080, min_value=100, max_value=2000)
                dpg.add_slider_int(label="X pos", tag=SC_X, min_value=0, max_value=1900)
                dpg.add_slider_int(label="Y pos", tag=SC_Y, min_value=0, max_value=1200)
            dpg.add_image(SC_C)
    else:
        dpg.show_item(DIM_W)
        dpg.show_item(SC_W)


def apply_inplace_filters(dets) -> None:
    """
    Applies the thresh, game-specific, second-nms and grouping detection filters. All are in-place operations.

    :param dets: detections returned by net
    :return: nothing
    """

    global current_game

    under_thresh = time.time()
    filter_under_thresh(dets)
    print(f"under thresh time is {time.time() - under_thresh:.4f}s")
    bygame = time.time()
    filter_detections_by_game(current_game, dets)
    print(f"by game time is {time.time() - bygame:.4f}s")
    second_nms_time = time.time()
    second_nms(dets)
    print(f"second nms time is {time.time() - second_nms_time:.4f}s")
    fng = time.time()
    filter_non_group_detections(current_game, dets)
    print(f"non group time is {time.time() - fng:.4f}s")


def check_if_change_detections(dets) -> None:
    """
    Check if we should change the current detection for the player hand or cards pot. In case the same new hand has been
    detected X times in a row, we change the detection accordingly. The reasoning is to avoid noisy detections.

    :param dets: detections returned by the net
    :return: nothing
    """
    global last_player_hand, last_card_pot, same_cp_in_a_row, same_ph_in_a_row, DET_ROW_THRES, cards_pot, \
           player_hand, player_hand_labels, cards_pot_labels

    fcp, fph, last_card_pot, last_player_hand = compare_detections(last_card_pot, last_player_hand, cards_pot,
                                                                   player_hand, dets)
    same_ph_in_a_row = 1 if fph < 0 else same_ph_in_a_row + 1
    same_cp_in_a_row = 1 if fcp < 0 else same_cp_in_a_row + 1

    if same_ph_in_a_row >= DET_ROW_THRES:
        player_hand_labels = last_player_hand
        # if same_ph_in_a_row == DET_ROW_THRES:
        #     print(f"I think hand is {[pos_cls_inverse[l] for l in player_hand_labels]}")
    if same_cp_in_a_row >= DET_ROW_THRES:
        cards_pot_labels = last_card_pot
        # if same_cp_in_a_row == DET_ROW_THRES:
        #     print(f"I think card pot is {[pos_cls_inverse[l] for l in cards_pot_labels]}")


@torch.inference_mode()
def update_selected_window(model: torch.nn.Module):
    global selected_window_hwnd, img_normalized, last_camera_update, UPDATE_DETECTIONS_RATE, dets, CR_PROC_Y, CR_PROC_X, CR_PROC_HH, CR_PROC_WH,\
    current_game, cards_pot, player_hand

    if not dpg.does_item_exist(CR_PROC_TEXT):
        return
    full_time = time.time()
    w = dpg.get_value(CR_PROC_WH) if dpg.get_value(CR_PROC_WH) > 100 else 100
    h = dpg.get_value(CR_PROC_HH) if dpg.get_value(CR_PROC_HH) > 100 else 100
    x = dpg.get_value(CR_PROC_X)
    y = dpg.get_value(CR_PROC_Y)
    grab_time = time.time()
    img, _ = grab_selected_window_contents(hwnd=selected_window_hwnd, w=w, h=h, x=x, y=y)
    print(f"grabbing time is {time.time() - grab_time:.4f}s")
    additional_stuff = time.time()
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA) # needs to be RGBA for dearpygui
    img_tensor = torch.from_numpy(img[:, :, :3])
    # img = np.asarray(Image.fromarray(img).resize((FRCNN_W, FRCNN_H))) # resize to net dimensions
    img_tensor = img_tensor.permute(2, 0, 1)
    print(f"prep time is {time.time() - additional_stuff:.4f}s")
    shape = img_tensor.size()[1:]
    # img_tensor = torch.from_numpy(np.asarray(T.ToPILImage()(img_tensor).resize((1900, 1080))))
    # img_tensor = img_tensor.permute(2, 0, 1)
    if time.time() * 1000 - last_camera_update > UPDATE_DETECTIONS_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        det_time = time.time()
        detections = model(img_tensor.unsqueeze(0)) # act as batch of 1 x img_tensor
        print(f"inference time is {time.time() - det_time:.4f} s.")
        dets = detections[0]
        filter_time = time.time()
        apply_inplace_filters(dets)
        print(f"in_place filter time is {time.time() - filter_time:.4f}s")
        cards_pot, player_hand = get_player_hand(current_game, dets)
        cards_pot, player_hand = filter_same_card(dets, cards_pot, player_hand)
        check_if_change_detections(dets)
        print(f"filtering time is {time.time() - filter_time:.4f}s")

    draw_time = time.time()
    img_pil = draw_detection(T.ToPILImage()(img_tensor), dets, cards_pot, player_hand)
    print(f"drawing time is {time.time() - draw_time:.4f}s")
    resize_time = time.time()
    img_pil = img_pil.resize((min(dpg.get_viewport_width(), FRCNN_W), min(dpg.get_viewport_height(), FRCNN_H)))
    print(f"resize time is {time.time() - resize_time:.4f}s")
    # img_pil = img_pil.resize(shape[::-1])
    copy_time = time.time()
    img = np.asarray(img_pil)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA) # needs to be RGBA for dearpygui
    print(f"copy time is {time.time() - copy_time:.4f}")
    # img = np.power(img, [1.2, 1.03, 1.0, 1.0])
    pygui_time = time.time()
    img = img.flatten().astype(np.float32)
    img_normalized = img / 255
    dpg.set_value(CR_PROC_TEXT, img_normalized)
    print(f"pygui time is {time.time() - pygui_time:.4f}s")
    print(f"full time is {time.time() - full_time:.4f}")
    print("-"*140)


def _restart_agent(sender, app_data, user_data):
    """
    Wrapper fun to recall the change game function.

    :param sender:
    :param app_data:
    :param user_data:
    :return:
    """
    global current_game

    print(f"I'm working")
    _change_game(sender, current_game, user_data)


def _change_active_window(sender, app_data, user_data):
    global CR_PROCESS, img_normalized, selected_window_title, selected_window_hwnd, CR_PROC_DIM, FRCNN_H, FRCNN_W, CR_PROC_X, CR_PROC_Y, CR_PROC_HH, CR_PROC_WH, \
           PLAYER_HAND, pleft, pright, ptop, pbot, CARDS_POT, cleft, cright, ctop, cbot, CR_PROC_AG_BUT

    if not dpg.does_item_exist(CR_PROCESS):
        with dpg.window(tag=CR_PROCESS, width=dpg.get_viewport_width(), height=dpg.get_viewport_height()):
            w = dpg.get_viewport_width()
            h = dpg.get_viewport_height()
            selected_window_title = app_data
            img, selected_window_hwnd = grab_selected_window_contents(wName=selected_window_title, w=w, h=h)
            imc = img.copy()
            # img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
            img = img.flatten().astype(np.float32)
            img_normalized = img / 255
            dpg.add_button(label="Restart Agent", tag=CR_PROC_AG_BUT, callback=_restart_agent)
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(w, h, img_normalized, tag=CR_PROC_TEXT)
            dpg.add_image(CR_PROC_TEXT)
        dpg.show_item(CR_PROCESS)
    if not dpg.does_item_exist(CR_PROC_DIM):
        with dpg.window(tag=CR_PROC_DIM):
            w = FRCNN_W
            h = FRCNN_H
            x = 0
            y = 0
            dpg.add_slider_int(label="Width", tag=CR_PROC_WH, default_value=w, min_value=100, max_value=2000)
            dpg.add_slider_int(label="Height", tag=CR_PROC_HH, default_value=h, min_value=100, max_value=2000)
            dpg.add_slider_int(label="X pos", tag=CR_PROC_X, default_value=x, min_value=0, max_value=1900)
            dpg.add_slider_int(label="Y pos", tag=CR_PROC_Y, default_value=y, min_value=0, max_value=1080)
        dpg.show_item(CR_PROC_DIM)


def _change_game(sender, app_data, user_data):
    global current_game, current_game_engine

    current_game = app_data
    if current_game == "Blackjack":
        with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_BIG_new_action_space_fixed.model", "rb") as f:
            bj_agent = pickle.load(f)
        current_game_engine = BlackjackEngine(bj_agent=bj_agent)


def update_agent():
    global current_game_engine, cards_pot, player_hand, dets

    if current_game_engine is None or "labels" not in dets:
        return
    labels = dets["labels"]
    player_labels = labels[player_hand].cpu().numpy()
    player_labels = [pos_cls_inverse[l] for l in player_labels]
    card_labels = labels[cards_pot].cpu().numpy()
    card_labels = [pos_cls_inverse[l] for l in card_labels]
    current_game_engine.update_detections(player_labels, card_labels)
    current_game_engine.act()


def update_window_names():
    win_names = grab_all_open_windows()
    dpg.configure_item(SEL_W, items=win_names)


def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        # dpg.add_button(label="Capture camera", width=MW_W, height=MW_H // 4, callback=_capture_camera)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4, callback=_get_screen_area)
        win_names = grab_all_open_windows()
        print(win_names)
        dpg.add_combo(label="Select window", tag=SEL_W, items=win_names, width=MW_W // 2,
                      callback=_change_active_window)
        dpg.add_combo(label="Select game", items=GAMES, callback=_change_game)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4, tag="Debug", callback=_change_debug_text, user_data="Debug")
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

