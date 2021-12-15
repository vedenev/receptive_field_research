from config import config
from utils.coordinates_utils import centred_coordinates
import numpy as np
import torch
from typing import Tuple
import constants


class MetricsMeasurer:

    def __init__(self,
                 is_save : bool = False,
                 save_path : str = constants.SAVE_METRICS_PATH,
                 save_period: int = 100,
                 n_steps_max: int = None):
        spot_measure_radius = config.accuracy_measurer.spot_measure_radius
        height = config.dataset.image_size
        width = height
        dX, dY = centred_coordinates(width, height)
        spot_measure_radius_squared = spot_measure_radius ** 2
        radii_squared = dX ** 2 + dY ** 2
        conditions = radii_squared <= spot_measure_radius_squared
        y_spot, x_spot = np.nonzero(conditions)
        self.y_spot = y_spot
        self.x_spot = x_spot
        self.loss_total: np.float32 = 0.0
        self.accuracy_total: np.float32 = 0.0
        self.n_correct_total: int = 0
        self.n_samples_total: int = 0
        self.n_steps: int = 0
        self.is_save = is_save
        if is_save:
            self.save_path = save_path
            self.save_period = save_period
            self.n_steps_max = n_steps_max
            self.data_to_save = np.zeros((n_steps_max, 3), np.float32)

    def reset(self):
        self.loss_total = 0.0
        self.accuracy_total = 0.0
        self.n_correct_total = 0
        self.n_samples_total = 0
        self.n_steps = 0

    def process_batch(self,
                      is_diaeresis: torch.Tensor,
                      prediction: torch.Tensor,
                      loss: torch.Tensor) -> Tuple[np.float32, np.float32, int]:
        prediction_np = prediction.cpu().detach().numpy()
        is_diaeresis_np = is_diaeresis.numpy()
        loss_np = loss.cpu().detach().numpy()
        batch_size = prediction_np.shape[0]
        n_correct: int = 0
        for within_batch_index in range(batch_size):
            is_diaeresis_sample = is_diaeresis_np[within_batch_index]
            prediction_sample = prediction_np[within_batch_index, :, :, :]
            no_diaeresis_mean = \
                self.__mean_spot_value(prediction_sample[0, :, :])
            diaeresis_mean = \
                self.__mean_spot_value(prediction_sample[1, :, :])
            is_diaeresis_predict = diaeresis_mean >= no_diaeresis_mean
            if is_diaeresis_predict == is_diaeresis_sample:
                n_correct += 1

        accuracy = np.float32(n_correct) / batch_size
        if self.is_save:
            self.data_to_save[self.n_steps, :] = \
                [self.n_steps, loss_np, accuracy]
            save_condition_1 = self.n_steps % self.save_period == 0
            save_condition_2 = self.n_steps >= (self.n_steps_max - 1)
            save_condition = save_condition_1 or save_condition_2
            if save_condition:
                np.save(self.save_path, self.data_to_save)

        self.loss_total += loss_np
        self.n_correct_total += n_correct
        self.n_samples_total += batch_size
        self.n_steps += 1

        return loss_np, accuracy, batch_size

    def get_total_result(self) -> Tuple[np.float32, np.float32, int]:
        accuracy_total = np.float32(self.n_correct_total)
        accuracy_total /= self.n_samples_total
        self.loss_total /= self.n_samples_total
        return self.loss_total, accuracy_total, self.n_samples_total

    def __mean_spot_value(self, featuremap: np.ndarray) -> np.float32:
        in_spot = featuremap[self.x_spot, self.y_spot]
        mean_value = np.mean(in_spot)
        return mean_value