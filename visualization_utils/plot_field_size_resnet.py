import matplotlib.pyplot as plt
import numpy as np

def plot_field_size_resnet() -> None:
    DATA_PATH_BASE = 'experiments/results_archive'
    DATA_PATH_FILE = 'experiment_field_size_vs_depth_resnet50_2021_12_30.npy'

    DATA_PATH = DATA_PATH_BASE + '/' + DATA_PATH_FILE
    data = np.load(DATA_PATH)

    distances = data[:, 0]
    accuracies = data[:, 1]

    plt.plot(distances, accuracies, 'k.-')
    plt.xlabel('distance')
    plt.ylabel('accuracy')
    plt.title(DATA_PATH_FILE)
    plt.show()