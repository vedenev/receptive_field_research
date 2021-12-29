from train import Trainer
from torchvision.models.segmentation import fcn_resnet50
from dataset_generator import ESymbolDataset
from train import PostTrainEvaluator
import numpy as np
import constants
from utils import JobTimer
import torch


def experiment_field_size_resnet50():
    distances = np.arange(7, 48 + 1)
    RESULT_SAVE_BASE_FILENAME = 'experiment_field_size_vs_depth_resnet50.npy'
    save_path = constants.SAVE_EXPERIMENTS_RESULTS_DIR \
        + '/' + RESULT_SAVE_BASE_FILENAME
    n_trains = distances.size
    save_data = np.zeros((distances.size, 2), np.float32)
    train_index = 0
    job_timer = JobTimer()
    for distance_index in range(distances.size):
        distance = distances[distance_index]
        dataset = ESymbolDataset(distance=distance, )
        net = fcn_resnet50(num_classes=constants.N_SYMBOLS,
                           pretrained=False)
        net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                             padding=3, bias=False)
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


