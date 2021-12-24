import matplotlib.pyplot as plt
import numpy as np

def plot_field_size_by_forward_pass() -> None:
    DATA_PATH_BASE = 'experiments/results_archive'
    DATA_PATH_FILE = 'experiment_field_size_by_forward_pass_fro_shifted_2021_12_24.npz'
    DATA_PATH = DATA_PATH_BASE + '/' + DATA_PATH_FILE
    npzfile = np.load(DATA_PATH)
    depthes = npzfile['depthes']
    radial_x_centred = npzfile['radial_x_centred']
    save_data = npzfile['save_data']
    depth_1 = np.min(depthes)
    depth_2 = np.max(depthes)
    radial_1 = np.min(radial_x_centred)
    radial_2 = np.max(radial_x_centred)
    plt.figure()
    extent = [depth_1 - 0.5, depth_2 + 0.5, radial_1 - 0.5, radial_2 + 0.5]
    plt.imshow(save_data, extent=extent, origin='lower')
    plt.xlabel('depth')
    plt.ylabel('radius')
    x = np.linspace(0, depth_2, 300)
    y = 3 * np.sqrt(x)
    plt.plot(x, y, 'r-', label='y = 3 * sqrt(x)')
    plt.title('output' + ', ' + DATA_PATH_FILE)
    plt.legend(loc='upper left')
    plt.colorbar()

    plt.figure()
    x_2 = np.arange(depth_2)
    y_2 = 3 * np.sqrt(x_2)
    y_2 = np.round(y_2).astype(np.int64)
    level = save_data[y_2, x_2]
    plt.plot(x_2, level, 'k.-')
    plt.xlabel('depth')
    plt.ylabel('level')

    plt.show()