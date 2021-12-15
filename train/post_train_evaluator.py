import torch
from config import config
import numpy as np
from typing import Tuple
from train import Trainer
from train import MetricsMeasurer

class PostTrainEvaluator:
    def __init__(self, trainer: Trainer = None):
        self.net = trainer.net
        self.loss_function = trainer.loss_function
        self.data_iterator = trainer.data_iterator
        self.device = trainer.device
        self.n_samples = config.post_train_evaluator.n_samples
        self.measurer = MetricsMeasurer(is_save=False)



    def __call__(self) -> Tuple[np.float32, np.float32]:
        self.net.eval()
        sample_index: int = 0
        with torch.no_grad():
            while sample_index < self.n_samples:
                batch = next(self.data_iterator)
                image = batch["image"]
                image = image.to(self.device)
                image_mask = batch["image_mask"]
                image_mask = image_mask.to(self.device)
                is_diaeresis = batch["is_diaeresis"]
                prediction = self.net(image)

                loss_batch = self.loss_function(prediction, image_mask)
                loss_value, accuracy, batch_size = \
                    self.measurer.process_batch(is_diaeresis,
                                                prediction,
                                                loss_batch)
                sample_index += batch_size
        loss_total, accuracy_total, _ = self.measurer.get_total_result()
        return loss_total, accuracy_total



