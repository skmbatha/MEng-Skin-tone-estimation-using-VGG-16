"""This class contains wrappers for C functions that process
data"""

import ctypes as c
import pathlib
from ctypes import *
from ctypes import POINTER as P
import numpy as np

class ImageUtils:

    @staticmethod
    def mod_lab_range(image):
        image=image.reshape(224*224*3,1)

        libname=pathlib.Path().absolute() /"preprocess_data/image_utils/image_utils_lib.so"
        lib=c.CDLL(str(libname))

        #Define function prototype
        lib.mod_lab_range.argtypes=POINTER(c.c_float),c.c_uint32
        lib.mod_lab_range.restype=None

        #Cast input file to a C types type
        image=image.astype(np.float32)
        image_arg=image.ctypes.data_as(POINTER(c.c_float)) 

        #run data conversion
        lib.mod_lab_range(image_arg,c.c_uint32(224*224))
        
        #Get converted data
        image=np.ctypeslib.as_array(image_arg,shape=[224,224,3])   

        return image

    @staticmethod
    def randomise_pixels(image, seed = 0):
    
        libname=pathlib.Path().absolute() /"preprocess_data/image_utils/image_utils_lib.so"
        lib=c.CDLL(str(libname))

        #Define function prototype
        lib.randomise_pixels_3.argtypes=c.c_uint32,c.c_uint32,c.c_uint32,P(c.c_float),P(c.c_float),P(c.c_float),
        P(c.c_float),P(c.c_float),P(c.c_float),P(c.c_float),P(c.c_float),P(c.c_float)
        lib.randomise_pixels_3.restype=None

        #Cast input file to a C types type
        image=image.astype(np.float32)

        #Convert all sub-arrays into pointers
        img_0=image[0].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_1=image[1].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_2=image[2].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_3=image[3].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_4=image[4].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_5=image[5].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_6=image[6].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_7=image[7].reshape(224*224).ctypes.data_as(P(c.c_float))
        img_8=image[8].reshape(224*224).ctypes.data_as(P(c.c_float))

        #Run data conversion
        channels=9
        array_length=224*224
        lib.randomise_pixels_3 (
            c.c_uint32(channels),
            c.c_uint32(array_length),
            c.c_uint32(seed), 
            img_0,img_1,img_2,img_3,img_4,img_5,img_6,img_7,img_8
            )
        
        #Get converted data
        image[0]=np.ctypeslib.as_array(img_0,shape=[224,224])
        image[1]=np.ctypeslib.as_array(img_1,shape=[224,224])
        image[2]=np.ctypeslib.as_array(img_2,shape=[224,224])
        image[3]=np.ctypeslib.as_array(img_3,shape=[224,224])
        image[4]=np.ctypeslib.as_array(img_4,shape=[224,224])
        image[5]=np.ctypeslib.as_array(img_5,shape=[224,224])
        image[6]=np.ctypeslib.as_array(img_6,shape=[224,224])
        image[7]=np.ctypeslib.as_array(img_7,shape=[224,224])
        image[8]=np.ctypeslib.as_array(img_8,shape=[224,224])

        return image

