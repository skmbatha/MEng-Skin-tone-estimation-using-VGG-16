import os,sys,bz2
import torch
import time
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
from PIL import Image
from process_image import ImageUtils
from skimage.color import rgb2lab,rgb2hsv
from torch.utils.data import Dataset
from torchvision.io import read_image

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
 
    def __init__(self, annotations_file, img_dir, 
                 transform=None, 
                 target_transform=None,
                 show_images=False,
                 randomise_pixels=False,
                 pack_9d={'v1':False,'v2':False},
                 contrast=1,
                 normalise=True,
                 blur=0
                 ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.show_images = show_images
        self.randomise_pixels = randomise_pixels
        self.pack_9d = pack_9d
        self.normalise = normalise
        self.contrast = contrast
        self.blur = blur
    
    def __data__(self):
        return self.img_labels
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem_disabled__(self, idx):
        """This is a __getitem__ function for reading and processing binary image information
        that was pre-randomised using pickle"""

        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        #read 9D image
        #f=bz2.BZ2File(img_path,'rb')
        f=open(img_path,'rb')
        image=pickle.load(f)
        f.close()

        #---------------------------------------------------------------
        #Visualise data
        if self.show_images:
            self.display_v2_in_out_binary_images(image)
            sys.exit()


        #================================================================
        NORM=[255, 255, 255, 179, 255, 255, 100, 220, 220]
        MEAN=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        #STD=[0.229, 0.224, 0.22,0,0,0,0.200, 0.224, 0.225]

        #normalise
        norm=lambda x,N:x/N
        for c in range(0,9):
            image[c]=norm(image[c],NORM[c])

        #center
        data_mean=np.mean(image.reshape(9,224*224),axis=0)
        mean=lambda x,m,M:x+(M-m)
        for c in range(0,9):
            image[c]=mean(image[c],data_mean[c],MEAN[c])

        #Variance
        #...

        label = torch.tensor(label, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)

        return image, label

    def __getitem__(self, idx):
        """ This is the original __getitem__ function. It reads any normal regression dataset with images
        and then transform into 9D"""

        label = self.img_labels.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        org_image = cv.imread(img_path)

        #Apply contrast
        """if self.contrast!=1:
            org_image =self.set_contrast(org_image,self.contrast)
        copy_org_image = org_image"""

        #Apply a gaussian blur
        """"if self.blur!=0:
            org_image =self.apply_blur(org_image,self.blur)"""

        #Make RGB,HSC,LAB(modded)
        rgb_image=np.array(org_image)
        hsv_image=rgb2hsv(np.array(org_image))
        lab_image=ImageUtils.mod_lab_range(rgb2lab(np.array(org_image)))

        #Merge fragments : Previous implementation that gave good results : V1
        if self.pack_9d['v1']:
            image=np.stack((rgb_image.reshape(3,224,224),hsv_image.reshape(3,224,224),lab_image.reshape(3,224,224)),axis=1)
            image=image.reshape(9,224,224)

        #Merge fragments in [R,G,B,H,S,V,L,A,B] format : V2
        if self.pack_9d['v2']:
            rgb_image=np.transpose(rgb_image.reshape(224*224,3)).reshape(3,224,224)
            hsv_image=np.transpose(hsv_image.reshape(224*224,3)).reshape(3,224,224)
            lab_image=np.transpose(lab_image.reshape(224*224,3)).reshape(3,224,224)
            image=np.array(list(rgb_image)+list(hsv_image)+list(lab_image))

        #Apply pixel randomisation
        """if self.randomise_pixels:
            image=ImageUtils.randomise_pixels(image,seed=0)"""

        #---------------------------------------------------------------
        #Visualise data
        """if self.show_images:
            if self.blur:
                self.display_blur_in_out(image,copy_org_image,rgb_image)
            if self.pack_9d['v1']:
                self.display_v1_in_out(image,copy_org_image)
            if self.pack_9d['v2']:
                self.display_v2_in_out(image,copy_org_image)
            sys.exit()"""

        #================================================================
        NORM=[255, 255, 255, 179, 255, 255, 100, 220, 220]
        MEAN=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        #STD=[0.229, 0.224, 0.22,0,0,0,0.200, 0.224, 0.225]

        #normalise
        norm=lambda x,N:x/N
        for c in range(0,9):
            image[c]=norm(image[c],NORM[c])

        #center
        data_mean=np.mean(image.reshape(9,224*224),axis=0)
        mean=lambda x,m,M:x+(M-m)
        for c in range(0,9):
            image[c]=mean(image[c],data_mean[c],MEAN[c])

        #Variance
        #...

        label = torch.tensor(label, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)

        return image, label
    
    #-------------------------------------------------

    def apply_blur(self,img,kernel_size):
        """This blur function uses a 2d convolution.
           Kernel size (1,IMG_DIM)"""
        img = cv.blur(img,(kernel_size,kernel_size))
        return img
    
    def set_contrast(self,img,contrast=1):
        """Decrease contrast (0,1), increase contrast (1,inf)"""
        img = cv.convertScaleAbs(img, alpha=contrast)
        return img
    
    def display_v2_in_out(self,image,org_image):
        img=np.array([
            image[0].astype(np.uint8).reshape(224,224),
            image[1].astype(np.uint8).reshape(224,224),
            image[2].astype(np.uint8).reshape(224,224)
            ])
        
        #Display transformed image
        cv.imshow("Random pic",np.concatenate((org_image,np.transpose(img)),axis=1)) #for V1
        cv.waitKey(0) # waits until a key is pressed
        cv.destroyAllWindows() # destroys the window showing image

    def display_v2_in_out_binary_images(self,image):
        img=np.array([
            image[0].astype(np.uint8).reshape(224,224),
            image[1].astype(np.uint8).reshape(224,224),
            image[2].astype(np.uint8).reshape(224,224)
            ])
        
        #Display transformed image
        cv.imshow("Random pic",np.transpose(img)) #for V2
        cv.waitKey(0) # waits until a key is pressed
        cv.destroyAllWindows() # destroys the window showing image


    def display_v1_in_out(self,image,org_image):
        img=np.array([
            image[0].astype(np.uint8).reshape(224,224),
            image[3].astype(np.uint8).reshape(224,224),
            image[6].astype(np.uint8).reshape(224,224)
            ])
        
        #Display transformed image
        cv.imshow("Random pic",np.concatenate((org_image,np.transpose(img)),axis=1)) #for V1
        cv.waitKey(0) # waits until a key is pressed
        cv.destroyAllWindows() # destroys the window showing image

    def display_blur_in_out(self,image,org_image,rgb_image):
        img=np.array([
            image[0].astype(np.uint8).reshape(224,224),
            image[1].astype(np.uint8).reshape(224,224),
            image[2].astype(np.uint8).reshape(224,224)
            ])
        
        if self.pack_9d['v2']:
            cv.imshow("Random pic",np.concatenate((org_image,np.transpose(rgb_image),np.transpose(img)),axis=1)) #for V2
        elif self.pack_9d['v1']:
            cv.imshow("Random pic",np.concatenate((org_image,rgb_image.reshape(224,224,3),np.transpose(img)),axis=1)) #for V2

        cv.waitKey(0) # waits until a key is pressed
        cv.destroyAllWindows() # destroys the window showing image
