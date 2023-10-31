import sys
import os,shutil
import cv2 as cv

DATASET_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\raw data\\1-285"
HEAD_ROTATIONS=['front-facing','left-facing','right-facing','up-facing','down-facing']

if __name__ =="__main__":

    images_dir=os.listdir(DATASET_DIR)
    for img_dir in images_dir:
        try:
            shutil.move(f"{DATASET_DIR}\\{img_dir}",f"{DATASET_DIR}\\{int(float(img_dir))-1}_")
        except:
            pass

    images_dir=os.listdir(DATASET_DIR)
    for img_dir in images_dir:
        shutil.move(f"{DATASET_DIR}\\{img_dir}",f"{DATASET_DIR}\\{img_dir.replace('_','')}")
