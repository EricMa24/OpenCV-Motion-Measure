from typing import *
import numpy as np


# Data Types
ColorImage = np.ndarray[(..., ..., 3), np.int8]
GrayImage = np.ndarray[(..., ..., 1), np.int8]
Image = Union[ColorImage, GrayImage]
