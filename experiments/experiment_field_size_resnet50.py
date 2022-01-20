from train import Trainer
from nets import get_resnet_50_adapted
from dataset_generator import ESymbolDataset
from train import PostTrainEvaluator
import numpy as np
import constants
from utils import JobTimer


RESULT_SAVE_BASE_FILENAME = 'experiment_field_size_vs_depth_resnet50.npy'


def experiment_field_size_resnet50() -> None:
    distances = np.arange(8, 110 + 1, 2)

    save_path = constants.SAVE_EXPERIMENTS_RESULTS_DIR \
        + '/' + RESULT_SAVE_BASE_FILENAME
    n_trains = distances.size
    save_data = np.zeros((distances.size, 2), np.float32)
    train_index = 0
    job_timer = JobTimer()
    for distance_index in range(distances.size):
        distance = distances[distance_index]
        dataset = ESymbolDataset(distance=distance, )
        net = get_resnet_50_adapted()
        trainer = Trainer(dataset=dataset, net=net)
        trainer()
        net = trainer.net

        evaluator = PostTrainEvaluator(trainer=trainer)
        loss, accuracy = evaluator()
        print('distance =', distance,
              ': ',
              'loss =', loss,
              'accuracy =', accuracy)
        save_data[train_index, :] = [distance, accuracy]
        np.save(save_path, save_data)
        train_index += 1
        job_done_fraction = (distance_index + 1) / distances.size
        job_timer(job_done_fraction)


