import numpy as np
import pandas as pd
import cv2 as cv
from random import random
import os,math,shutil,sys,json
from crop import find_crop_face

#GLOBAL MACROS
INPUT_DIRECTORY="D:\\Users\\Public\\Documents\\raw_data_2_train\\1-285"
OUTPUT_DIRECTORY_NAME="dataset_eye_cropped_1"
OUTPUT_DIRECTORY="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\regr_separate_identities"


#GLOBAL DATA STRUCTURES
CONFIG=[
    {
        "category":"training",
        "size":60,
        "data":[]
    },
    {
        "category":"validation",
        "size":40,
        "data":[]
    },
    {
        "category":"test",
        "size":0,
        "data":[]
    }
]

if sum([x['size'] for x in CONFIG])>100:
    raise Exception("Dataset partitions don't add up to 100%")

#CREATE RANDOM PARTITIONS
raw_images=os.listdir(INPUT_DIRECTORY)
num_images=len(raw_images)

for cat in CONFIG:
    for _ in range(round(cat['size']*0.01*num_images)):
        i=math.floor(random()*len(raw_images))
        cat['data'].append(raw_images[i])
        raw_images.pop(i)

# CREATE DATASET
for cat in CONFIG:

    annotations_txt=""

    #Create folders
    DIR_1=f"{OUTPUT_DIRECTORY}\\{OUTPUT_DIRECTORY_NAME}"
    DIR_2=f"{DIR_1}\\{cat['category']}\\data"
    if os.path.exists(DIR_2):
        shutil.rmtree(DIR_2)
    os.makedirs(DIR_2)

    #COPY INPUT DATA TO OUTPUT + PRE-PROCESSING
    for identity in cat['data']:

        #GET IDENTITY IMAGES
        DIR=f"{INPUT_DIRECTORY}\\{identity}\\front-facing"
        identity_data=os.listdir(DIR)
        identity_data=list(filter(lambda x:x!="cropped",identity_data))
        identity_data=list(filter(lambda x:'jpeg' in x or 'jpg' in x,identity_data))
        identity_data=list(filter(lambda x:'_200.' not in x,identity_data))

        # GET IMAGE LIST
        images=[]
        for img_label in identity_data:
            images.append({'name':img_label,'image':cv.imread(f"{INPUT_DIRECTORY}\\{identity}\\front-facing\\{img_label}")})

        #SAVE CROPPPED IMAGES TO OUTPUT FOLDER        
        for image in images:
            try:
                cv.imwrite(f"{DIR_2}\\{image['name']}", image['image'])
            except Exception as e:
                print("Error during image write")

        #READ LABEL & ADD IMAGES TO ANNOTATION STRUCT
        f=open(f"{INPUT_DIRECTORY}\\{identity}\\monk_scale_value.json",'r')
        label=json.loads(f.read())["value"]
        f.close()
        for image_name in identity_data:
            annotations_txt+=f"{image_name},{label}\n"


    #SAVE ANNOTATIONS FILE
    annotations_file=f"{DIR_1}\\{cat['category']}\\annotations.csv"
    f=open(annotations_file,'w')
    f.write(annotations_txt)
    f.close()
