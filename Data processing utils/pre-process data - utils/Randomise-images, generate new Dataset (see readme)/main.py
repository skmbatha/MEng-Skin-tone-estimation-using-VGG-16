import os
import bz2
import cv2
import shutil
import pickle
import pandas as pd
from randomise import apply_randomise_pixels

#GLOBAL MACROS
DATASET_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\regr_separate_identities\\dataset_1"
OUTPUT_DIR_NAME="regr_randomised_binary_images"
OUTPUT_DIRECTORY="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data"

CONFIG=[
    {
        'category':'training',
        'seed':{'start':0,'end':5}
    },
    {
        'category':'validation',
        'seed':{'start':6,'end':10}
    }
]

#ENTRY POINT
if __name__ == "__main__":


    for cat in CONFIG:

        #Create folders
        DIR_1=f"{OUTPUT_DIRECTORY}\\{OUTPUT_DIR_NAME}"
        DIR_2=f"{DIR_1}\\{cat['category']}\\data"
        if os.path.exists(DIR_2):
            shutil.rmtree(DIR_2)
        os.makedirs(DIR_2)

        #READ IMAGE NAMES
        images=os.listdir(f"{DATASET_DIR}\\{cat['category']}\\data")

        #GET LABELS
        ref_ann=pd.read_csv(f"{DATASET_DIR}\\{cat['category']}\\annotations.csv")
        input_data=[{'name':v[1],'label':v[2]} for v in ref_ann.itertuples()]
        input_data_size=len(input_data)

        #REMOVE MISSING IMAGES
        annotations_txt=""
        for seed in range(cat['seed']['start'],cat['seed']['end']):
            print(f"Generating images for SEED ({cat['category']}): {seed}")
            for ann in input_data:
                if ann['name'] in images:

                    #Apply randomisation
                    img = apply_randomise_pixels(f"{DATASET_DIR}\\{cat['category']}\\data\\{ann['name']}", seed)

                    #Save file as ND array
                    #f=bz2.BZ2File(f"{OUTPUT_DIRECTORY}\\{OUTPUT_DIR_NAME}\\{cat['category']}\\data\\seed_{seed}_{ann['name']}",'wb')
                    f=open(f"{OUTPUT_DIRECTORY}\\{OUTPUT_DIR_NAME}\\{cat['category']}\\data\\seed_{seed}_{ann['name']}",'wb')
                    pickle.dump(img,f)
                    f.close()

                    #add in new annotation
                    annotations_txt+=f"seed_{seed}_{ann['name']},{ann['label']}\n"

        #CREATE NEW ANNOTATIONS
        annotations_file=f"{OUTPUT_DIRECTORY}\\{OUTPUT_DIR_NAME}\\{cat['category']}\\annotations.csv"
        f=open(annotations_file,'w')            
        f.write(annotations_txt)
        f.close()

print("DONE 100%")