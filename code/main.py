from GUI.main_window import create_main_window, create_dpg_env, CAM_W, update_camera, update_screen_area
from Windows_utils.windows import grab_all_open_windows
import dearpygui.dearpygui as dpg
import threading

def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    while dpg.is_dearpygui_running():
        update_camera()
        update_screen_area()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

if __name__ == "__main__":
    main()