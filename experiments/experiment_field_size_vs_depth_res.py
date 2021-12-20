from train import Trainer
from dataset_generator import ESymbolDataset
from nets import NoPoolsNetRes
from train import PostTrainEvaluator
import numpy as np
import constants
from utils import JobTimer


def experiment_field_size_vs_depth_res():
    distances = np.arange(7, 48 + 1)
    depthes = np.arange(2, 48 + 1)
    RESULT_SAVE_BASE_FILENAME = 'experiment_field_size_vs_depth_res.npy'
    save_path = constants.SAVE_EXPERIMENTS_RESULTS_DIR\
                + '/' + RESULT_SAVE_BASE_FILENAME
    n_trains = distances.size * depthes.size
    save_data = np.zeros((n_trains, 3), np.float32)
    train_index = 0
    for distance_index in range(distances.size):
        distance = distances[distance_index]
        job_timer = JobTimer()
        for depth_index in range(depthes.size):
            depth = depthes[depth_index]
            dataset = ESymbolDataset(distance=distance)
            net = NoPoolsNetRes(depth=depth)
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


