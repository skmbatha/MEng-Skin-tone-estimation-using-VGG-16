"""This class contains wrappers for C functions that process
data"""

import ctypes as c
import pathlib
from ctypes import *
import numpy as np

class ImageUtils:

    @staticmethod
    def mod_lab_range(image):
        image=image.reshape(224*224*3,1)

        libname=pathlib.Path().absolute() /"preprocess_data/image_utils/image_utils.so"
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

