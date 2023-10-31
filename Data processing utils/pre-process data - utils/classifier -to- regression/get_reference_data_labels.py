import os
import json
import math
import matplotlib.pyplot as plt

csv_text=''
annotations_csv="annotations.csv"

root="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\1-290"
dirs=os.listdir(root)

data_count=0

for dir in dirs:
    try:
        f=open(f"{root}\\{dir}\\monk_scale_value.json",'r')
        data=f.read()
        f.close()
        try:
            #read the monk skin value
            monk=float(json.loads(data)["value"])

            #read the front facing image labels
            images=os.listdir(f"{root}\\{dir}\\front-facing\\cropped")

            #save in csv
            for img in images:
                csv_text+=f"{img},{monk}\n"

            #count data
            data_count+=1
            print(f"Loaded data {data_count}\n")
        except:
            print("No value at {}".format(dir))
            continue
            
    except Exception as e:
        print(e)
        print("Problem reading monk_skin_value.json")
        pass

#create reference csv file and write output
f=open("ref_annotations.csv",'w')
f.write(csv_text)
f.close()