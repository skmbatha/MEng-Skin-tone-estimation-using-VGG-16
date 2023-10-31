import os
import shutil
import pandas as pd
from random import random


INPUT_DIR="input_data\\pad_50_2"
OUTPUT_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\regression_data\\random_sets"
OUTPUT_FOLDER_NAME="pad_50_data_6"
OUTPUT_DATA_DIR=f"{OUTPUT_DIR}\\{OUTPUT_FOLDER_NAME}"
OUTPUT_TRAIN_DIR=f"{OUTPUT_DATA_DIR}\\training"
OUTPUT_VALIDATION_DIR=f"{OUTPUT_DATA_DIR}\\validation"
OUTPUT_TRAIN_DATA_DIR=f"{OUTPUT_TRAIN_DIR}\\data"
OUTPUT_VALIDATION_DATA_DIR=f"{OUTPUT_VALIDATION_DIR}\\data"
TRAINING_PERCENTAGE=70

if __name__ == "__main__":

    #Create output folders
    try:
        print(f"Create folders @ {OUTPUT_DATA_DIR}")
        if os.path.exists(OUTPUT_DATA_DIR):
            shutil.rmtree(OUTPUT_DATA_DIR)
        os.makedirs(OUTPUT_DATA_DIR)
        os.makedirs(f"{OUTPUT_TRAIN_DATA_DIR}")
        os.makedirs(f"{OUTPUT_VALIDATION_DATA_DIR}")
    except Exception as e:
        print(e)

    #Read the input dataset
    ref_ann=pd.read_csv(f"{INPUT_DIR}\\annotations.csv")
    input_data=[{'name':v[1],'label':v[2]} for v in ref_ann.itertuples()]
    input_data_size=len(input_data)

    #Calc proportions
    training_data_size=int((TRAINING_PERCENTAGE/100)*input_data_size)
    validation_data_size=input_data_size-training_data_size
    print(f"Creating data TRAIN({training_data_size}): VAL({validation_data_size})")

    #output files
    validation=[]

    #Create validation data
    print("Copying validation images...")
    N=input_data_size
    while N!=training_data_size:
        r=int(random()*N)
        data_point=input_data[r]
        validation.append(data_point)
        try:
            shutil.copyfile(f"{INPUT_DIR}\\images\\{data_point['name']}", f"{OUTPUT_VALIDATION_DATA_DIR}\\{data_point['name']}")
        except Exception as e:
            print(e)
        input_data.pop(r)
        N=len(input_data)
    
    #Create training data
    print("Copying training images...")
    for i in range(0,len(input_data)):
        data_point=input_data[i]
        try:
            shutil.copyfile(f"{INPUT_DIR}\\images\\{data_point['name']}", f"{OUTPUT_TRAIN_DATA_DIR}\\{data_point['name']}")
        except Exception as e:
            print(e)

    #Create annotation files
    print(f"Creating validation csv, data_length: {len(validation)}")
    csv_text=""
    for img in validation:
        csv_text+=f"{img['name']},{img['label']}\n"
    val_file=open(f"{OUTPUT_VALIDATION_DIR}\\annotations.csv",'w')
    val_file.write(csv_text)
    val_file.close()

    print(f"Creating training csv, data_length: {len(input_data)}")
    csv_text=""
    for img in input_data:
        csv_text+=f"{img['name']},{img['label']}\n"
    val_file=open(f"{OUTPUT_TRAIN_DIR}\\annotations.csv",'w')
    val_file.write(csv_text)
    val_file.close()

    #Validate the input data & validation
    counted_sims=0
    for tr_entry in input_data:
        for entry in validation:
            if entry['name']==tr_entry['name']:
                counted_sims+=1
    print(f"Found {counted_sims} overlaps in the val vs train dataset")

    