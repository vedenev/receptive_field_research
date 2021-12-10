import numpy as np
import cv2
from config import config
from typing import Tuple

class Augmentator:
    def __init__(self, image_shape: Tuple[int, int] = (128, 128)):
        self.amplitude = config.dataset.augmentation.amplitude
        kernel_size = config.dataset.augmentation.size
        self.kernel_sizes = (kernel_size, kernel_size)
        self.image_shape = image_shape
        x = np.arange(image_shape[1], dtype=np.float32)
        y = np.arange(image_shape[0], dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        self.X = X
        self.Y = Y


    def __call__(self, image: np.ndarray) -> np.ndarray:
        shifts_X = self.__generate_random_smooth_image()
        shifts_Y = self.__generate_random_smooth_image()
        X_map = self.X + shifts_X
        Y_map = self.Y + shifts_Y
        image = cv2.remap(image, X_map, Y_map, cv2.INTER_CUBIC)
        return image


    def __generate_random_smooth_image(self):
        random_map = \
            np.random.uniform(-1.0, 1.0, self.image_shape)
        random_map = random_map.astype(np.float32)
        random_map = cv2.blur(random_map, self.kernel_sizes)
        random_map /= np.max(random_map)
        random_map *= self.amplitude
        return random_map



