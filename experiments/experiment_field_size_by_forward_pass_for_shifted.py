from nets import NoPoolsNetRes
import numpy as np
import constants
from config import config
import torch

RESULT_SAVE_BASE_FILENAME = \
    'experiment_field_size_by_forward_pass_fro_shifted.npz'


def experiment_field_size_by_forward_pass_for_shifted() -> None:
    depthes = np.arange(1, 64 + 1)

    save_path = constants.SAVE_EXPERIMENTS_RESULTS_DIR\
                + '/' + RESULT_SAVE_BASE_FILENAME
    image_size = config.by_forward_pass.image_size
    center = image_size // 2
    radial_y = np.arange(0, center)
    radial_y_centred = radial_y
    radial_x = center
    input_image = np.zeros((1, 1, image_size, image_size), np.float32)
    input_image[0, 0, center, center] = 1.0
    input_image = torch.from_numpy(input_image)
    save_data = np.zeros((radial_y.size, depthes.size), np.float32)
    for depth_index in range(depthes.size):
        if depth_index % 10 == 0:
            print("depth_index:", depth_index, depthes.size)
        depth = depthes[depth_index]
        net = NoPoolsNetRes(depth=depth, is_shifted_init=True)
        net.eval()
        output_image = net(input_image)
        radial = output_image[0, 0, 0: center, radial_x]
        radial_np = radial.cpu().detach().numpy()
        radial_np = radial_np[::-1]
        radial_np = np.abs(radial_np)
        radial_np = radial_np / np.max(radial_np)
        save_data[:, depth_index] = radial_np

    np.savez(save_path,
             depthes=depthes,
             radial_x_centred=radial_y_centred,
             save_data=save_data)



