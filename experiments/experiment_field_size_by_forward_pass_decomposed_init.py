from nets import NoPoolsNetRes
import numpy as np
import constants
from config import config
import torch
from initializers import decomposed_init

RESULT_SAVE_BASE_FILENAME = 'experiment_field_size_by_forward_pass_decomposed_init.npz'


def experiment_field_size_by_forward_pass_decomposed_init() -> None:
    depthes = np.arange(2, 24 + 1, 2)

    save_path = constants.SAVE_EXPERIMENTS_RESULTS_DIR\
                + '/' + RESULT_SAVE_BASE_FILENAME
    image_size = config.by_forward_pass.image_size
    center = image_size // 2
    radial_x = np.arange(center, image_size)
    radial_x_centred = radial_x - center
    radial_y = center
    input_image = np.zeros((1, 1, image_size, image_size), np.float32)
    input_image[0, 0, center, center] = 1.0
    input_image = torch.from_numpy(input_image)
    save_data = np.zeros((radial_x.size, depthes.size), np.float32)
    for depth_index in range(depthes.size):
        if depth_index % 10 == 0:
            print("depth_index:", depth_index, depthes.size)
        depth = depthes[depth_index]
        net = NoPoolsNetRes(depth=depth, is_shifted_init=False)
        decomposed_init(net)
        net.eval()
        output_image = net(input_image)
        radial = output_image[0, 0, radial_y, radial_x[0]: radial_x[-1] + 1]
        radial_np = radial.cpu().detach().numpy()
        radial_np = np.abs(radial_np)
        radial_np = radial_np / np.max(radial_np)
        save_data[:, depth_index] = radial_np

    np.savez(save_path,
             depthes=depthes,
             radial_x_centred=radial_x_centred,
             save_data=save_data)



