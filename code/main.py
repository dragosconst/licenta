from GUI.main_window import create_main_window, create_dpg_env, update_camera, update_screen_area, update_selected_window,\
                            update_agent, update_window_names
import dearpygui.dearpygui as dpg
from Models.Vision.faster import train_frcnn, validate, train_fccnn_reference, get_faster
from Utils.utils import load_dataloader, get_loader


def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    frcnn = get_faster("../data/frcnn_resnet50_5k_per_class_smol_e1.pt")
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
