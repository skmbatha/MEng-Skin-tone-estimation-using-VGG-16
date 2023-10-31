import os,sys
import torch
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
from process_image import ImageUtils
from skimage.color import rgb2lab
from torch.utils.data import Dataset
from torchvision.io import read_image
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil

class RegrDataset_(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path) 
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label
    
class RegrDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None,lab=True,color_correct=True,**kwargs):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.conv_to_lab_space = lab
        self.color_correct=color_correct


        if kwargs['blur']:
            print("Image blurring activated!")
            self.blur=True
        else:
            self.blur=False
    
    def __data__(self):
        return self.img_labels
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path) 

        # Color correction, white balancing
        # https://pypi.org/project/colorcorrect/
        """if(self.color_correct):
            image=to_pil(cca.grey_world(from_pil(image)))"""

        if(self.blur):
            image = cv.blur(from_pil(image) ,(224,224))
            image=to_pil(image)
        
        #conv from rgb to Lab if requested
        if(self.conv_to_lab_space):
            image = rgb2lab(np.array(image))

            #Convert a,b to range (0-220), won't go outside rgb value ranges
            #---------------------------------------------------------------
            #Running this on a bare metal layer(C): 18.54(same as if it didn't exist)
            image=ImageUtils.mod_lab_range(image)

            #Pure Python : ~30s (train)
            """image=image.reshape(224*224,3)
            for Pixel in image:
                Pixel[1]==110
                Pixel[2]==110
            image=image.reshape(224,224,3)"""

            #Using Pandas (hope it's C accelerated) : 49.72s
            """image=image.reshape(224*224,3)
            df = pd.DataFrame(image,columns=['L', 'a','b'])
            df.loc[:,'a'] +=110
            df.loc[:,'b'] +=110 
            image=df.values.tolist()
            image=np.array(image).reshape(224,224,3)"""
            #---------------------------------------------------------------

            #convert to PIL
            image=Image.fromarray(np.uint8(image))

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label