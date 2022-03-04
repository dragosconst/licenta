
from torch.optim import Adam
import torch.utils.data
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from GUI.main_window import create_main_window, create_dpg_env, update_camera, update_screen_area
import dearpygui.dearpygui as dpg
from Models.VIsion.faster import train_frcnn, initialize_frcnn
from Utils.utils import load_dataloader, get_loader

def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    while dpg.is_dearpygui_running():
        update_camera()
        update_screen_area()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

if __name__ == "__main__":
    # main()
    torch.cuda.empty_cache()
    # backbone = torchvision.models.vgg19(pretrained=True).features
    # backbone.out_channels = 512

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 128, 512),), aspect_ratios=((0.5, 1.0, 2.0), ))

    roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    frcnn = initialize_frcnn(backbone=backbone, cls=55, anchor_gen=anchor_generator, roi_pooler=roi_pool)
    frcnn.to("cuda")
    dataset, dataloader = load_dataloader(batch_size=1)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 1/5), int(len(dataset) * 4/5)])
    train_loader = get_loader(train_set, batch_size=1)
    valid_loader = get_loader(val_set, batch_size=1)

    adam = Adam(frcnn.parameters(), lr=0.001)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(adam, [10, 20, 25], gamma=0.1)

    train_frcnn(frcnn, adam, lr_scheduler=lr_sched, train_dataloader=train_loader, valid_dataloader=valid_loader,
                device="cuda", num_epochs=30)