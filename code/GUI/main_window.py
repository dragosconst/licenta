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
GAMES = ["Blackjack", "Poker Texas Hold'Em", "Razboi", "Macao", "Septica", "Solitaire"]
current_game = None # type: str
current_game_engine = None # type: BaseEngine

# image and net specific constants
# --------------------------------
img_normalized = None
video_capture = None
UPDATE_DETECTIONS_RATE = 500 # ms interval at which to update our detections, used to avoid slowing down the program too much
last_camera_update = 0
dets = {} # type: Dict[str, torch.Tensor]
selected_window_title = None
selected_window_hwnd = None
same_ph_in_a_row = 0
same_cp_in_a_row = 0
last_player_hand = []
last_card_pot = []
DET_ROW_THRES = 3
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
    global last_camera_update, UPDATE_DETECTIONS_RATE, dets

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


def apply_inplace_filters(dets) -> None:
    """
    Applies the thresh, game-specific, second-nms and grouping detection filters. All are in-place operations.

    :param dets: detections returned by net
    :return: nothing
    """

    global current_game

    filter_under_thresh(dets)
    filter_detections_by_game(current_game, dets)
    second_nms(dets)
    filter_non_group_detections(current_game, dets)


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
    w = dpg.get_value(CR_PROC_WH) if dpg.get_value(CR_PROC_WH) > 100 else 100
    h = dpg.get_value(CR_PROC_HH) if dpg.get_value(CR_PROC_HH) > 100 else 100
    x = dpg.get_value(CR_PROC_X)
    y = dpg.get_value(CR_PROC_Y)
    img, _ = grab_selected_window_contents(hwnd=selected_window_hwnd, w=w, h=h, x=x, y=y)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA) # needs to be RGBA for dearpygui
    img = np.asarray(Image.fromarray(img).resize((FRCNN_W, FRCNN_H))) # resize to net dimensions
    img_tensor = torch.from_numpy(img[:, :, :3])
    img_tensor = img_tensor.permute(2, 0, 1)
    shape = img_tensor.size()[1:]
    # img_tensor = torch.from_numpy(np.asarray(T.ToPILImage()(img_tensor).resize((1900, 1080))))
    # img_tensor = img_tensor.permute(2, 0, 1)
    if time.time() * 1000 - last_camera_update > UPDATE_DETECTIONS_RATE: # update every UCR ms the bounding boxes
        last_camera_update = time.time() * 1000

        img_tensor = T.ConvertImageDtype(torch.float32)(img_tensor)
        img_tensor = img_tensor.to("cuda")
        detections = model(img_tensor.unsqueeze(0)) # act as batch of 1 x img_tensor
        dets = detections[0]
        apply_inplace_filters(dets)
        cards_pot, player_hand = get_player_hand(current_game, dets)
        cards_pot, player_hand = filter_same_card(dets, cards_pot, player_hand)
        check_if_change_detections(dets)


    img_pil = draw_detection(T.ToPILImage()(img_tensor), dets, cards_pot, player_hand)
    img_pil = img_pil.resize((dpg.get_viewport_width(), dpg.get_viewport_height()))
    # img_pil = img_pil.resize(shape[::-1])
    img[:dpg.get_viewport_height(), :dpg.get_viewport_width(), :3] = np.asarray(img_pil)
    # img = np.power(img, [1.2, 1.03, 1.0, 1.0])
    img = img.flatten().astype(np.float32)
    img_normalized = img / 255
    dpg.set_value(CR_PROC_TEXT, img_normalized)


def _change_active_window(sender, app_data, user_data):
    global CR_PROCESS, img_normalized, selected_window_title, selected_window_hwnd, CR_PROC_DIM, FRCNN_H, FRCNN_W, CR_PROC_X, CR_PROC_Y, CR_PROC_HH, CR_PROC_WH, \
           PLAYER_HAND, pleft, pright, ptop, pbot, CARDS_POT, cleft, cright, ctop, cbot

    if not dpg.does_item_exist(CR_PROCESS):
        with dpg.window(tag=CR_PROCESS, width=FRCNN_W, height=FRCNN_H):
            w = FRCNN_W
            h = FRCNN_H
            selected_window_title = app_data
            img, selected_window_hwnd = grab_selected_window_contents(wName=selected_window_title, w=w, h=h)
            imc = img.copy()
            # img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
            img = img.flatten().astype(np.float32)
            img_normalized = img / 255
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
        with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_BIG_state_replay.model", "rb") as f:
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


def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        dpg.add_button(label="Capture camera", width=MW_W, height=MW_H // 4, callback=_capture_camera)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4, callback=_get_screen_area)
        win_names = grab_all_open_windows()
        print(win_names)
        dpg.add_combo(label="Select window", tag=SEL_W, items=win_names, width=MW_W // 2,
                      callback=_change_active_window)

        dpg.add_combo(label="Select game", items=GAMES, callback=_change_game)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4, tag="Debug", callback=_change_debug_text, user_data="Debug")
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

