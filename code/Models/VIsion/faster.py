import torch
from torchvision.models import vgg19
from torchvision.models.detection.rpn import AnchorGenerator


"""
I will use a different RPN head for the Faster R-CNN network, for better fine tuning of the convolutional layer.
I will also use a different backbone, probably a pre-trained ImageNet network.
"""