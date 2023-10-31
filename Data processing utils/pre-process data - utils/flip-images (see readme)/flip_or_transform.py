import os
import cv2
import pandas as pd

"""
cv2.ROTATE_90_CLOCKWISE
cv2.ROTATE_180
cv2.ROTATE_90_COUNTERCLOCKWISE
"""

#GLOBAL MACROS
DATASET_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\regr_separate_identities\\dataset_eye_cropped_1_aug\\validation"
PREFIX="rot_180"

#ENTRY POINT
if __name__ == "__main__":

    #READ IMAGE NAMES
    images=os.listdir(f"{DATASET_DIR}\\data")

    #GET LABELS
    ref_ann=pd.read_csv(f"{DATASET_DIR}\\annotations.csv")
    input_data=[{'name':v[1],'label':v[2]} for v in ref_ann.itertuples()]
    input_data_size=len(input_data)

    #REMOVE MISSING IMAGES
    annotations_txt=""
    for ann in input_data:
        if ann['name'] in images:

            #read image, flip and save a new  annotation entry
            img = cv2.imread(f"{DATASET_DIR}\\data\\{ann['name']}")
            #img = cv2.flip(img, 1)
            img = cv2.rotate(img,cv2.ROTATE_180)
            cv2.imwrite(f"{DATASET_DIR}\\data\\{PREFIX}_{ann['name']}", img)

            #add in new annotation
            annotations_txt+=f"{ann['name']},{ann['label']}\n"
            annotations_txt+=f"{PREFIX}_{ann['name']},{ann['label']}\n"

    #CREATE NEW ANNOTATIONS
    annotations_file=f"{DATASET_DIR}\\annotations.csv"
    f=open(annotations_file,'w')            
    f.write(annotations_txt)
    f.close()

print("DONE 100%")