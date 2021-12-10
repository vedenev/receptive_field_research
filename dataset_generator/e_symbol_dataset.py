import constants
import numpy as np
import cv2
from config import config
from dataset_generator.augmentator import Augmentator
from torch.utils.data import Dataset
import typing
import torch


#  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class ESymbolDataset(Dataset):
    def __init__(self, distance: int = 8, size: int = 128):
        self.size = size
        e_symbol = cv2.imread(constants.E_SYMBOL_PATH, cv2.IMREAD_GRAYSCALE)
        diaeresis = cv2.imread(constants.DIAERESIS_PATH, cv2.IMREAD_GRAYSCALE)

        margin = int(np.ceil(config.dataset.augmentation.amplitude))
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
        self.image_base_without_diaeresis = np.copy(image_base)


        # add diaeresis:
        x1 = center - diaeresis.shape[1] // 2
        x2 = x1 + diaeresis.shape[1]
        y1 = center - diaeresis.shape[0] // 2 - distance
        assert y1 >= 0, "y1 < 0, distance too big"
        y2 = y1 + diaeresis.shape[0]
        image_base[y1: y2, x1: x2] = diaeresis

        self.image_base_with_diaeresis = image_base


        self.augmentator = Augmentator(image_shape=image_shape)

        x = np.arange(size).astype(np.float32)
        y = np.arange(size).astype(np.float32)
        X, Y = np.meshgrid(x, y)
        center_x = np.float32(size) / 2
        center_y = np.float32(size) / 2
        dX = X - center_x
        dY = Y - center_y
        spot_size_squared = np.float32(config.dataset.spot_size) ** 2
        image_mask_base = np.exp(-(dX ** 2 + dY ** 2) / spot_size_squared)
        image_mask_base = self.__numpy_to_torch_tensor(image_mask_base)
        self.image_mask_base = image_mask_base

        self.image_mask_shape = (constants.N_SYMBOLS, size, size)

    def __len__(self) -> int:
        return 2**48  # a big number, because infinite dataset size

    def __getitem__(self, idx_not_used: typing.Any) -> \
            typing.Dict[str, torch.Tensor]:
        is_diaeresis = np.random.rand() < constants.DIAERESIS_PROBABILITY
        image_mask = np.zeros(self.image_mask_shape, np.float32)
        if is_diaeresis:
            image = self.augmentator(self.image_base_with_diaeresis)
            image_mask[1, :, :] = self.image_mask_base
        else:
            image = self.augmentator(self.image_base_without_diaeresis)
            image_mask[0, :, :] = self.image_mask_base
        image = image[self.margin: -self.margin, self.margin: -self.margin]
        image = self.__to_float32(image)
        image = self.__numpy_to_torch_tensor(image)
        image_mask = torch.from_numpy(image_mask)
        return {'image': image,
                'image_mask': image_mask}

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

