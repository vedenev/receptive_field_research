import numpy as np
from typing import Tuple

def centred_coordinates(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(width).astype(np.float32)
    y = np.arange(height).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    center_x = np.float32(width) / 2
    center_y = np.float32(height) / 2
    dX = X - center_x
    dY = Y - center_y
    return dX, dY