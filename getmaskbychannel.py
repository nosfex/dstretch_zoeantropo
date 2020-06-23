
from loadimage import load_image_channel
from loadimage import load_image_raw
import PIL
from PIL import Image
import pathlib
import numpy as np

def get_mask_by_mask_index(filename, mask_index, scale_factor):
    npdata = load_image_channel(filename).astype(np.float)

    result = npdata[...,:3]

    for i in range(0,3): 
        if i == mask_index: 
            result[...,i] *= scale_factor
        else:
            result[...,i] *= 0
  
    return result.astype(np.uint8)