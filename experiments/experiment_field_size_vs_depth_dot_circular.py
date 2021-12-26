from train import Trainer
from dataset_generator import ESymbolDotDataset
from nets import NoPoolsNetRes
from train import PostTrainEvaluator
import numpy as np
import constants
from utils import JobTimer
from initializers import circular_init


def experiment_field_size_vs_depth_dot_circular():
    distances = np.arange(7, 24 + 1, 2)
    depthes = np.arange(3, 48 + 1, 2)
    RESULT_SAVE_BASE_FILENAME = 'experiment_field_size_vs_depth_dot_180.npy'
    save_path = constants.SAVE_EXPERIMENTS_RESULTS_DIR\
                + '/' + RESULT_SAVE_BASE_FILENAME
    n_trains = distances.size * depthes.size
    save_data = np.zeros((n_trains, 3), np.float32)
    train_index = 0
    job_timer = JobTimer()
    for distance_index in range(distances.size):
        distance = distances[distance_index]
        for depth_index in range(depthes.size):
            depth = depthes[depth_index]
            dataset = ESymbolDotDataset(distance=distance)
            net = NoPoolsNetRes(depth=depth, is_shifted_init=False)
            circular_init(net)
            trainer = Trainer(dataset=dataset, net=net)
            trainer()
            net = trainer.net

            evaluator = PostTrainEvaluator(trainer=trainer)
            loss, accuracy = evaluator()
            print('distance =', distance,
                  'depth =', depth,
                  ': ',
                  'loss =', loss,
                  'accuracy =', accuracy)
            save_data[train_index, :] = [distance, depth, accuracy]
            np.save(save_path, save_data)
            train_index += 1
        job_done_fraction = (distance_index + 1) / distances.size
        job_timer(job_done_fraction)


