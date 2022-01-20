import matplotlib.pyplot as plt
import numpy as np

DATA_PATH_BASE = 'experiments/results_archive'
DATA_PATH_FILE = \
    'experiment_field_size_by_forward_pass_decomposed_init_2021_12_23.npz'
DATA_PATH = DATA_PATH_BASE + '/' + DATA_PATH_FILE


def plot_field_size_by_forward_pass_decomposed_init() -> None:
    npzfile = np.load(DATA_PATH)
    depthes = npzfile['depthes']
    radial_x_centred = npzfile['radial_x_centred']
    save_data = npzfile['save_data']
    depth_1 = np.min(depthes)
    depth_2 = np.max(depthes)
    radial_1 = np.min(radial_x_centred)
    radial_2 = np.max(radial_x_centred)
    extent = [depth_1 - 0.5, depth_2 + 0.5, radial_1 - 0.5, radial_2 + 0.5]
    plt.imshow(save_data, extent=extent, origin='lower')
    plt.xlabel('depth')
    plt.ylabel('radius')
    x = np.linspace(0, depth_2, 300)
    y = x
    plt.plot(x, y, 'r-', label='y = x')
    plt.title('output' + ', ' + DATA_PATH_FILE)
    plt.legend(loc='upper left')
    plt.colorbar()

    plt.show()
