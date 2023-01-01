import numpy as np
from PIL import Image


def png_to_array(path: str) -> np.ndarray:
    image = Image.open(path)
    arr = np.asarray(image)
    return arr
