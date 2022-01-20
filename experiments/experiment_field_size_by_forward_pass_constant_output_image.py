from nets import NoPoolsNet
import numpy as np
from config import config
import torch
import matplotlib.pyplot as plt


def experiment_field_size_by_forward_pass_constant_output_image() -> None:
    image_size = config.by_forward_pass.image_size
    center = image_size // 2
    input_image = np.zeros((1, 1, image_size, image_size), np.float32)
    input_image[0, 0, center, center] = 1.0
    input_image = torch.from_numpy(input_image)
    depth = 24
    net = NoPoolsNet(depth=depth, is_constant_init=True)
    net.eval()
    output_image = net(input_image)
    output_image = output_image[0, 0, :, :]
    output_image_np = output_image.cpu().detach().numpy()

    plt.imshow(output_image_np)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('output')
    plt.colorbar()
    plt.show()




