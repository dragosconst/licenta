from GUI.main_window import create_main_window, create_dpg_env, update_screen_area, update_selected_window,\
                            update_agent, update_window_names
import dearpygui.dearpygui as dpg
from Models.Vision.faster import get_faster


def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    # frcnn = get_faster("../data/frcnn_resnet50_posind_e2.pt")
    frcnn = get_faster("../data/frcnn_resnet50_betterrpn_e1.pt")
    # frcnn = get_faster("../data/frcnn_resnet50_5k_per_class_stretched_e0.pt")
    # frcnn = get_faster("../data/frcnn_resnet50_5k_per_class_smol_e3_no_noise.pt")
    # frcnn = get_faster("../data/frcnn_mobilenetv3_large_posind_e7.pt", mobilenet=True)
    frcnn.eval()
    while dpg.is_dearpygui_running():
        update_window_names()
        update_screen_area(frcnn)
        update_selected_window(frcnn)
        update_agent()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
