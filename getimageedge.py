
from PIL import Image, ImageFilter
import loadimage
import pathlib
import numpy as np

def get_edge_simple(filename):
    image = loadimage.load_image_raw(filename)

    new_image = image.filter(ImageFilter.FIND_EDGES)
    return np.array(new_image)