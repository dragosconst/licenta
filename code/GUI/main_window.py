from Windows_utils.windows import grab_all_open_windows, grab_selected_window_contents
import dearpygui.dearpygui as dpg
import cv2 as cv

# window names
MAIN_WINDOW = "Main"
SEL_W = "selectw"
DEBUG_STATE = True
MW_W = 1000
MW_H = 630
FS = 60
CR_PROCESS = "cr_proc"
CR_PROC_TEXT = "cr_proc_text"

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

def _change_active_window(sender, app_data, user_data):
    global CR_PROCESS
    if not dpg.does_item_exist(CR_PROCESS):
        with dpg.window(tag=CR_PROCESS):
            print("huh")
            img = grab_selected_window_contents(app_data)
            cv.imshow("img", img[...,:3])
            cv.waitKey(0)
            cv.destroyAllWindows()
            w, h, c, data = dpg.load_image("debug.bmp")
            with dpg.texture_registry(show=True):
                # tx_id = dpg.add_raw_texture(800, 600, grab_selected_window_contents(app_data).flatten(),
                #                             tag=CR_PROC_TEXT, format=dpg.mvFormat_Float_rgba)
                dpg.add_raw_texture(800, 600, data, tag="tx_id")
            print("finished")
            dpg.add_image("tx_id")
        dpg.show_item(CR_PROCESS)
        print("hello?")



def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        win_names = grab_all_open_windows()
        print(win_names)
        dpg.add_combo(label="Select window", tag=SEL_W, items=win_names, width=MW_W//2, callback=_change_active_window)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Select game", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4, tag="Debug", callback=_change_debug_text, user_data="Debug")
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

