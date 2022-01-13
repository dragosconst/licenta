from GUI.main_window import create_main_window, create_dpg_env
import dearpygui.dearpygui as dpg

def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()