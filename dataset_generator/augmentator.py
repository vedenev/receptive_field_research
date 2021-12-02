import numpy as np
import cv2
from config import config

class Augmentator:
    def __init__(self):
        self.amplitude = config.augmentation.amplitude
        self.size = config.augmentation.size

    def __call__(self, image:  np.ndarray) -> np.ndarray:
        height, width = image.shape
        self.shape = (height, width)
        shifts_x = self.__generate_random_smooth_image()
        shifts_y = self.__generate_random_smooth_image()

    def __generate_random_smooth_image(self):
        random_map = \
            np.random.uniform(-1.0, 1.0, self.shape)
        random_map = random_map.astype(np.float32)



