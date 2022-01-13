from GUI.main_window import create_main_window, create_dpg_env, SEL_W
from Windows_utils.windows import grab_all_open_windows
import dearpygui.dearpygui as dpg

def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    while dpg.is_dearpygui_running():
        win_names = grab_all_open_windows()
        dpg.configure_item(SEL_W, items=win_names)
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

if __name__ == "__main__":
    main()