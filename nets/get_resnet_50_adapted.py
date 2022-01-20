import torch
from torchvision.models.segmentation import fcn_resnet50
import constants


def get_resnet_50_adapted() -> torch.nn.Module:
    net = fcn_resnet50(
        pretrained=False,
        progress=True,
        num_classes=constants.N_SYMBOLS,
    )

    net.backbone.conv1 = torch.nn.Conv2d(constants.N_INPUT_CHANNELS,
                                         64,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=False)

    return net

