import sys
import os,shutil
import cv2 as cv

DATASET_DIR="D:\\Users\\Public\\Documents\\raw_data_2_train\\1-285"
HEAD_ROTATIONS=['front-facing','left-facing','right-facing','up-facing','down-facing']

if __name__ =="__main__":

    counter=1
    images_dir=sorted(os.listdir(DATASET_DIR))
    for img_dir in images_dir:
        try:
            shutil.move(f"{DATASET_DIR}\\{img_dir}",f"{DATASET_DIR}\\{counter}_")
            counter+=1
        except:
            pass
        
    images_dir=sorted(os.listdir(DATASET_DIR))
    for img_dir in images_dir:
        shutil.move(f"{DATASET_DIR}\\{img_dir}",f"{DATASET_DIR}\\{img_dir.replace('_','')}")
