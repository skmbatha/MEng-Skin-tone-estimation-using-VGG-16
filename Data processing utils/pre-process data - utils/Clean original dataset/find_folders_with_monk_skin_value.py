import os,shutil,json
import cv2 as cv

DATASET_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\raw data\\1-285"
HEAD_ROTATIONS=['front-facing','left-facing','right-facing','up-facing','down-facing']
MONK_IN_SEARCH=6

if __name__ =="__main__":

    images_dir=os.listdir(DATASET_DIR)
    for img_dir in images_dir:
        f=open(f"{DATASET_DIR}\\{img_dir}\\monk_scale_value.json",'r')
        data=f.read()
        f.close()
        value=json.loads(data)
        monk=float(value["value"])

        if monk==MONK_IN_SEARCH:
            print(f"Found : {DATASET_DIR}\\{img_dir}")
