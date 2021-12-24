import constants
import numpy as np
import cv2
from config import config
from dataset_generator.augmentator import Augmentator
from torch.utils.data import Dataset
import typing
import torch
from utils.coordinates_utils import centred_coordinates


#  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class ESymbolDotDataset(Dataset):
    def __init__(self, distance: int = 8):
        self.distance = distance
        image_size = config.dataset.image_size
        e_symbol = cv2.imread(constants.E_SYMBOL_PATH, cv2.IMREAD_GRAYSCALE)
        self.diaeresis = cv2.imread(constants.DIAERESIS_DOT_PATH,
                               cv2.IMREAD_GRAYSCALE)

        margin = int(np.ceil(config.dataset.augmentation.amplitude))
        self.margin = margin
        size_margined = image_size + 2 * margin
        self.size_margined = size_margined
        image_shape = (size_margined, size_margined)
        image_base = np.full(image_shape, 255, np.uint8)
        center = size_margined // 2
        self.center = center

        # add e symbol:
        x1 = center - e_symbol.shape[1] // 2
        x2 = x1 + e_symbol.shape[1]
        y1 = center - e_symbol.shape[0] // 2
        y2 = y1 + e_symbol.shape[0]
        image_base[y1: y2, x1: x2] = e_symbol
        self.image_base_without_diaeresis = image_base

        self.augmentator = Augmentator(image_shape=image_shape)

        dX, dY = centred_coordinates(image_size, image_size)
        spot_size_squared = np.float32(config.dataset.spot_size) ** 2
        image_mask_base = np.exp(-(dX ** 2 + dY ** 2) / spot_size_squared)
        image_mask_base = self.__numpy_to_torch_tensor(image_mask_base)
        self.image_mask_base = image_mask_base

        self.image_mask_shape = (constants.N_SYMBOLS, image_size, image_size)

    def __len__(self) -> int:
        return 2**48  # a big number, because infinite dataset size

    def __getitem__(self, idx_not_used: typing.Any) -> \
            typing.Dict[str, torch.Tensor]:
        is_diaeresis = np.random.rand() < constants.DIAERESIS_PROBABILITY
        image_mask = np.zeros(self.image_mask_shape, np.float32)
        if is_diaeresis:
            # add diaeresis:
            image = np.copy(self.image_base_without_diaeresis)
            angle = config.dataset.dot_angle_range_degrees * np.pi / 180
            angle_random = 3 * np.pi / 2 + angle * (np.random.rand() - 0.5)
            center_x = self.center + self.distance * np.cos(angle_random)
            center_x = int(np.round(center_x))
            center_y = self.center + self.distance * np.sin(angle_random)
            center_y = int(np.round(center_y))
            x1 = center_x - self.diaeresis.shape[1] // 2
            assert x1 >= 0, "x1 < 0, distance too big"
            x2 = x1 + self.diaeresis.shape[1]
            assert x2 <= (self.size_margined - 1), \
                "x2 > (size_margined - 1), distance too big"
            y1 = center_y - self.diaeresis.shape[0] // 2
            assert y1 >= 0, "y1 < 0, distance too big"
            y2 = y1 + self.diaeresis.shape[0]
            assert y2 <= (self.size_margined - 1), \
                "y2 > (size_margined - 1), distance too big"
            image[y1: y2, x1: x2] = self.diaeresis

            image = self.augmentator(image)
            image_mask[1, :, :] = self.image_mask_base
        else:
            image = self.augmentator(self.image_base_without_diaeresis)
            image_mask[0, :, :] = self.image_mask_base
        image = image[self.margin: -self.margin, self.margin: -self.margin]
        image = self.__to_float32(image)
        image = self.__numpy_to_torch_tensor(image)
        image_mask = torch.from_numpy(image_mask)
        is_diaeresis_np = np.asarray(is_diaeresis)
        is_diaeresis_tensor = torch.from_numpy(is_diaeresis_np)
        return {'image': image,
                'image_mask': image_mask,
                'is_diaeresis': is_diaeresis_tensor}

    @staticmethod
    def __numpy_to_torch_tensor(image: np.ndarray) -> torch.Tensor:
        image = np.expand_dims(image, axis=0)  # torch image: C x H x W
        image = torch.from_numpy(image)
        return image

    @staticmethod
    def __to_float32(image: np.ndarray) -> np.ndarray:
        image = 255 - image
        image = image.astype(np.float32)
        image /= np.float32(255.0)
        return image

