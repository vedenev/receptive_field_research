import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def limits(x: np.ndarray) -> Tuple[np.int64, np.int64, np.int64, np.ndarray]:
    #min_ = np.min(x)
    min_ = 1
    max_ = np.max(x)
    size_ = max_ - min_ + 1
    range_ = np.arange(min_, max_ + 1)
    return min_, max_, size_, range_

def plot_field_size_vs_depth() -> None:
    DATA_PATH_BASE = 'experiments/results_archive'
    #DATA_PATH_FILE = 'experiment_field_size_vs_depth_2021_12_16_unfinished.npy'
    #DATA_PATH_FILE = 'experiment_field_size_vs_depth_res_2021_12_20.npy'
    #DATA_PATH_FILE = 'experiment_field_size_vs_depth_res_special_init_2021_12_21.npy'
    #DATA_PATH_FILE = 'experiment_field_size_vs_depth_res_2021_12_22.npy'
    DATA_PATH_FILE = 'experiment_field_size_vs_depth_res_decomposed_init_2021_12_23.npy'

    IS_LINEAR_PLOT = True

    DATA_PATH = DATA_PATH_BASE + '/' + DATA_PATH_FILE
    data = np.load(DATA_PATH)
    # data[train_index, :] = [distance, depth, accuracy]
    calculated = np.nonzero(data[:, 0] > 0)[0]
    data = data[calculated, :]
    distances = data[:, 0]
    distances = distances.astype(np.int64)
    depths = data[:, 1]
    depths = depths.astype(np.int64)
    accuracies = data[:, 2]

    distance_1, distance_2, distance_size, distance_range = limits(distances)
    depth_1, depth_2, depth_size, depth_range = limits(depths)
    accuracy_table = np.full((distance_size, depth_size), np.nan, np.float32)
    for train_index in range(data.shape[0]):
        distance = distances[train_index]
        distance_index = distance - distance_1
        depth = depths[train_index]
        depth_index = depth - depth_1
        accuracy = accuracies[train_index]
        accuracy_table[distance_index, depth_index] = accuracy

    extent = [depth_1 - 0.5, depth_2 + 0.5, distance_1 - 0.5, distance_2 + 0.5]
    plt.imshow(accuracy_table, extent=extent, origin='lower')
    plt.xlabel('depth')
    plt.ylabel('distance')
    x = np.linspace(0, depth_2, 300)
    if IS_LINEAR_PLOT:
        y = x
        plt.plot(x, y, 'r-', label='y = x')
    else:
        y = 3 * np.sqrt(x)
        plt.plot(x, y, 'r-', label='y = 3 * sqrt(x)')
    plt.title('accuracy' + ', ' + DATA_PATH_FILE)
    plt.legend(loc='lower right')
    plt.colorbar()
    plt.show()







