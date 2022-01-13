from Windows_utils.windows import grab_all_open_windows
import dearpygui.dearpygui as dpg

# window names
MAIN_WINDOW = "Main"
SEL_W = "selectw"
DEBUG_STATE = True
MW_W = 1000
MW_H = 630
FS = 60

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
    pass

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

