import os
import json
import math
import time
import shutil
import pandas as pd
import matplotlib.pyplot as plt

#input ref annotations
ref_annotations_csv="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\pre-process data - utils\\classifier -to- regression\\ref_annotations.csv"

#output annotations
csv_text=''
annotations_csv="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\brightness_2_3_aug_pad50_2\\validation\\annotations.csv"

#input "classifier" data folder
input_root="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\clustered data_2\\brightness_2_3_aug_pad50_2\\validation"
dirs=os.listdir(input_root)

#output directory
output_root='C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\brightness_2_3_aug_pad50_2\\validation'

data_count=0
tot_images=0
#read ref annotationa
ref_ann=pd.read_csv(ref_annotations_csv)

#create regression dataset
for dir in dirs:
    try:
        #read all images in class
        images=os.listdir(f"{input_root}\\{dir}")
        tot_images+=len(images)
        
        #lookup a ref monk value in "ref_annotations_csv"
        for img in images:
            found=False
            for v in ref_ann.itertuples():
                if v[1].strip() in img:
                    #copy image into data folder
                    shutil.copyfile(f"{input_root}\\{dir}\\{img}", f"{output_root}\\data\\{img}")

                    #append info to 'annotations_csv'
                    csv_text+=f"{img},{v[2]}\n"

                    #done
                    data_count+=1
                    #print(f"Loaded data {data_count}\n")
                    found=True
                    break

    except Exception as e:
        print(e)
        pass
            
print(f"TOTAL IMAGES LOADED : {tot_images}")

#create reference csv file and write output
f=open(annotations_csv,'w')
f.write(csv_text)
f.close()