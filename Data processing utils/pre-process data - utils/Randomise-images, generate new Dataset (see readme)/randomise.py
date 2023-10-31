import cv2 as cv
import os
import pandas as pd
import numpy as np
from process_image import ImageUtils
from skimage.color import rgb2lab,rgb2hsv


def apply_randomise_pixels(img_path,seed):

    org_image = cv.imread(img_path)

    #Make RGB,HSC,LAB(modded)
    rgb_image=np.array(org_image)
    hsv_image=rgb2hsv(np.array(org_image))
    lab_image=ImageUtils.mod_lab_range(rgb2lab(np.array(org_image)))

    #Merge fragments in [R,G,B,H,S,V,L,A,B] format : V2
    rgb_image=np.transpose(rgb_image.reshape(224*224,3)).reshape(3,224,224)
    hsv_image=np.transpose(hsv_image.reshape(224*224,3)).reshape(3,224,224)
    lab_image=np.transpose(lab_image.reshape(224*224,3)).reshape(3,224,224)
    image=np.array(list(rgb_image)+list(hsv_image)+list(lab_image))

    #Apply pixel randomisation
    image=ImageUtils.randomise_pixels(image,seed)

    return image
