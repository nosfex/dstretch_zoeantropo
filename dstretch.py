
import numpy as np
import cv2
from PIL import Image
import glob
from matplotlib import pyplot as plt
from loadimage import load_image_channel
import pathlib
from getmaskbychannel import get_mask_by_mask_index
from getimageedge import get_edge_simple
def deco_stretch(file, target_mean = None, target_sigma = None):
    data_input = file
    data_mean, data_std = cv2.meanStdDev(data_input )
    stretch = None
    pca_data = data_input.flatten()  #np.asarray(data_input).reshape(-1)
    print(pca_data)
    pca_obj, pca_eigen, eigen_values = cv2.PCACompute2(pca_data, mean=target_mean)

    eig_data_sigma = np.sqrt(eigen_values)
    print(eig_data_sigma)
    scale = np.diag(1/eig_data_sigma) 
    print(scale)
    if target_sigma is None:
        stretch = np.diag(data_std)
    else:
        stretch = np.diag(target_sigma)
    stretch = np.float32(stretch)

    repmat_data = cv2.repeat(np.transpose(pca_obj), pca_data.shape[0], 1)
    zmat_data = cv2.subtract(pca_data, repmat_data, dtype=cv2.CV_32F)
   
    if target_mean is not None:
        repmat_data = cv2.repeat(np.transpose(target_mean), pca_data.shape[0], 1)
    transformed = zmat_data * (np.transpose(pca_eigen) * scale * pca_eigen * stretch)
    transformed = cv2.add(transformed, repmat_data, dtype=cv2.CV_32F)

    dstr32f = transformed.reshape( data_input.shape[0], -1, data_input.shape[2])

    return dstr32f


mean = np.multiply(np.ones(shape=[1,1], dtype=np.float32), 120)
sigma = np.multiply(np.ones(shape=[3,1], dtype=np.float32), 50)
import tkinter
def run():
    root = tkinter.Tk()
    filez = tkinter.filedialog.askopenfilenames(parent = root, title = 'Choose Files')
    file_dir = root.tk.splitlist(filez)
    for k in file_dir:

        data_input = cv2.imread(k)
        data_convert = cv2.cvtColor(data_input, cv2.COLOR_BGR2LAB)
        
        dstrlab32f = deco_stretch(data_convert, mean, sigma)
        dstrlab8u = np.uint8(dstrlab32f)
        data_from8u = cv2.cvtColor(dstrlab8u, cv2.COLOR_LAB2BGR)
        dstrbgr32f = deco_stretch(data_convert, mean, sigma)
        dstrbgr8u = np.uint8(dstrbgr32f)

        r,g,b = (get_mask_by_mask_index(k, 0,4), get_mask_by_mask_index(k, 1,4), get_mask_by_mask_index(k, 2,4)) 
        edge = get_edge_simple(k)
        
        titles = ['original', 'pil_edge','r','g', 'b', 'lab_convert', 'lap_dstretch_mapped', 'lab_dstretch', 'lab_rgb', 'rgb_dstretch_mapped', 'rgb_dstretch']
        images = [ cv2.imread(k), edge, r,g,b,data_convert, dstrlab32f, dstrlab8u,data_from8u,  dstrbgr32f, dstrbgr8u]

        for i in range(0, np.size(titles) ):
            print(i)
            plt.subplot(3, 4, i+1)
            plt.imshow(cv2.cvtColor((images[i]), cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()