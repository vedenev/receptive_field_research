import matplotlib.pyplot as plt
import constants
import numpy as np

def plot_metrics():
    data_path = constants.SAVE_METRICS_PATH
    data = np.load(data_path)
    steps = data[:, 0]
    loss = data[:, 1]
    accuracy = data[:, 2]

    plt.subplot(2, 1, 1)
    plt.plot(steps, loss, 'k.-')
    plt.xlabel('steps')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(steps, accuracy, 'k.-')
    plt.xlabel('steps')
    plt.ylabel('accuracy')

    plt.show()