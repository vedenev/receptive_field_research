import constants
import numpy as np
import cv2
from config import config
from dataset_generator.augmentator import Augmentator

class SampleGenerator:
    def __init__(self, distance: int = 8, size: int = 128):
        self.size = size
        e_symbol = cv2.imread(constants.E_SYMBOL_PATH, cv2.IMREAD_GRAYSCALE)
        diaeresis = cv2.imread(constants.DIAERESIS_PATH, cv2.IMREAD_GRAYSCALE)

        margin = int(np.ceil(config.augmentation.amplitude))
        self.margin = margin
        size_margined = size + 2 * margin
        image_shape = (size_margined, size_margined)
        image_base = np.full(image_shape, 255, np.uint8)
        center = size_margined // 2

        # add e symbol:
        x1 = center - e_symbol.shape[1] // 2
        x2 = x1 + e_symbol.shape[1]
        y1 = center - e_symbol.shape[0] // 2
        y2 = y1 + e_symbol.shape[0]
        image_base[y1: y2, x1: x2] = e_symbol

        # add diaeresis:
        x1 = center - diaeresis.shape[1] // 2
        x2 = x1 + diaeresis.shape[1]
        y1 = center - diaeresis.shape[0] // 2 - distance
        assert y1 >= 0, "y1 < 0, distance too big"
        y2 = y1 + diaeresis.shape[0]
        image_base[y1: y2, x1: x2] = diaeresis

        self.image_base = image_base

        self.augmentator = Augmentator(image_shape=image_shape)

    def __call__(self) -> np.ndarray:
        image = self.augmentator(self.image_base)
        image = image[self.margin: -self.margin, self.margin: -self.margin]
        image_negative = 255 - image
        return image_negative



