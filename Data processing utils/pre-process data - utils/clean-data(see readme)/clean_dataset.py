import os
import pandas as pd

#GLOBAL MACROS
DATASET_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\merge_fine_tune\\test_lighting_1\\validation"

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
            annotations_txt+=f"{ann['name']},{ann['label']}\n"

    #CREATE NEW ANNOTATIONS
    annotations_file=f"{DATASET_DIR}\\annotations.csv"
    f=open(annotations_file,'w')            
    f.write(annotations_txt)
    f.close()

print("DONE 100%")