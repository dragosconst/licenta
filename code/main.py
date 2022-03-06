import time

from torch.optim import Adam, SGD
import torch.utils.data
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

from GUI.main_window import create_main_window, create_dpg_env, update_camera, update_screen_area
import dearpygui.dearpygui as dpg
from Models.Vision.faster import train_frcnn, initialize_frcnn, train_fccnn_reference
from Utils.utils import load_dataloader, get_loader

def main():
    main_font = create_dpg_env()
    create_main_window(main_font)
    while dpg.is_dearpygui_running():
        update_camera()
        update_screen_area()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

def run_frcnn():
    # torch.cuda.empty_cache()
    frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to("cuda")
    in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 55).to("cuda")

    dataset, dataloader = load_dataloader(batch_size=1)
    torch.manual_seed(1)
    val_set, train_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 1/5), int(len(dataset) * 4/5)])
    indices = torch.randperm(len(dataset)).tolist()
    train_set = torch.utils.data.Subset(dataset, indices[:-4100])
    val_set = torch.utils.data.Subset(dataset, indices[:-4100])

    train_loader = get_loader(train_set, batch_size=2)
    valid_loader = get_loader(val_set, batch_size=1, shuffle=False)

    params = [p for p in frcnn.parameters() if p.requires_grad]
    sgd = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_sched = torch.optim.lr_scheduler.MultiStepLR(adam, [10, 20, 25], gamma=0.1)
    lr_sched = torch.optim.lr_scheduler.StepLR(sgd, step_size=3, gamma=0.1)

    # train_frcnn(frcnn, adam, lr_scheduler=lr_sched, train_dataloader=train_loader, valid_dataloader=valid_loader,
    #             device="cuda", num_epochs=30)

    # frcnn.load_state_dict(torch.load("D:\\facultate stuff\\licenta\\data\\frcnn_resnet50.pt"))
    # frcnn.eval()
    # frcnn.to("cpu")
    # torch.manual_seed(time.time())
    # rand_img = torch.randint(high=4199,size=(1,)).item()
    # print(rand_img)
    # img, targets = dataset[rand_img]
    # imgs = [img]
    # with torch.inference_mode():
    #     print("start pred")
    #     pred = frcnn(imgs)
    #     print("stop pred")
    #     for idx, p in enumerate(pred):
    #         print(p)
    #         def show(imgs):
    #             if not isinstance(imgs, list):
    #                 imgs = [imgs]
    #             fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    #             for i, img in enumerate(imgs):
    #                 img = img.detach()
    #                 img = F.to_pil_image(img)
    #                 axs[0, i].imshow(np.asarray(img))
    #                 axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #
    #         drawn_boxes = draw_bounding_boxes(transforms.ConvertImageDtype(torch.uint8)(img), p["boxes"],
    #                                           colors="red")
    #         print(len(p["boxes"]))
    #         show(drawn_boxes)
    #         plt.show()
    #         print(targets)
    # train_fccnn_reference(frcnn, sgd, lr_scheduler=lr_sched, train_dataloader=train_loader, valid_dataloader=valid_loader,
    #             device="cuda", num_epochs=10)

    # torch.save(frcnn.state_dict(), "D:\\facultate stuff\\licenta\\data\\frcnn_resnet502112.pt")

if __name__ == "__main__":
    run_frcnn()