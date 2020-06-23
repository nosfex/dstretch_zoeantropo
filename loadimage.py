from PIL import Image
import os   
import numpy as np

def load_image_raw(filename):
    image = None
    if(os.path.exists(filename)):
        image = Image.open(filename)

    return image


def load_image_channel(filename):
    
    image = load_image_raw(filename)
    if image == None:
        return None
    return np.array(image)
