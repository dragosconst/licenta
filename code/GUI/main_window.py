import dearpygui.dearpygui as dpg

# window names
MAIN_WINDOW = "Main"
MW_W = 1000
MW_H = 630
FS = 60

def create_dpg_env():
    dpg.create_context()
    with dpg.font_registry():
        default_font =  dpg.add_font('../data/fonts/Blazma-X3eVP.ttf', FS)
    dpg.create_viewport(title="Smart Cards", width=MW_W, height=MW_H)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    return default_font

def create_main_window(font):
    with dpg.window(tag=MAIN_WINDOW):
        dpg.add_button(label="Select window", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Select screen area", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Select game", width=MW_W, height=MW_H // 4)
        dpg.add_button(label="Debug mode:ON", width=MW_W, height=MW_H // 4)
        dpg.bind_font(font)

    dpg.set_primary_window(MAIN_WINDOW, True)

