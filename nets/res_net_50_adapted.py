import torch
from torchvision.models.segmentation import fcn_resnet50
#  https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762/10
class FCNResNet50Adapted(fcn_resnet50):

    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        super(fcn_resnet50, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)