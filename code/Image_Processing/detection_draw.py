from typing import List, Optional, Dict, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch

from Data_Processing.Raw_Train_Data.raw import pos_cls_inverse

def draw_detection(img: Image.Image, detections: Dict[str, torch.Tensor], cards_pot: List[int], player_hand: List[int]) \
        -> Image.Image:
    """
    Draw detections outputed by model over original image.
    :param img: original image
    :param detections: detections output by model
    :return: image with detections drawn over
    """

    img_copy = img.copy()
    draw_obj = ImageDraw.Draw(img_copy)
    font = ImageFont.truetype("../data/fonts/NotoSerifCJKjp-Medium.otf", 15)
    for idx, (box, label, score) in enumerate(zip(detections["boxes"], detections["labels"], detections["scores"])):
        if idx in cards_pot:
            color = "blue"
        else:
            color = "red"
        x1, y1, x2, y2 = box.cpu().numpy()
        draw_obj.rectangle((x1, y1, x2, y2), fill=None, outline=color)
        draw_obj.rectangle((x1, y1 - 40, x1 + 60, y1), fill=color, outline=color)
        # draw_obj.arc(((x1+x2)//2-300,(y1+y2)//2-300,(x1+x2)//2+300, (y1+y2)//2+300), start=0, end=360, fill=color)
        draw_obj.text((x1, y1 - 40), f"{pos_cls_inverse[label.item()][:3]} {str(round(score.item() * 100, 2))}%", font=font, fill=(255, 255, 255, 255))
    return img_copy