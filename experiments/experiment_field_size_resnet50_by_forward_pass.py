import torch
import numpy as np
import matplotlib.pyplot as plt
from nets import get_resnet_50_adapted
from initializers import resnet_constant_init
from config import config


def experiment_field_size_resnet50_by_forward_pass() -> None:
    net = get_resnet_50_adapted()
    resnet_constant_init(net)

    size = config.dataset.image_size

    image = np.zeros((1, 1, size, size), np.float32)
    image[0, 0, size // 2, size // 2] = 1.0
    image = torch.from_numpy(image)

    output = net(image)
    out = output['out']
    out = out.cpu().detach().numpy()
    out = out[0, 0, :, :]
    out = out / np.max(out)

    plt.figure()
    plt.imshow(out)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('output, normalized')
    plt.colorbar()

    plt.figure()
    plt.plot(out[size // 2:, size // 2], 'k.-', label='output')
    level = 0.0012
    plt.plot([0, size], [level, level], 'r--', label=str(level) + ' level')
    plt.xlabel('radius')
    plt.ylabel('value')
    plt.legend()

    plt.show()
