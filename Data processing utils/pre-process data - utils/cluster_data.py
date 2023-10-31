import os
import json
import math
import shutil

abs_dir="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\1-290"
all_people=os.listdir(abs_dir)

for person in all_people:
    #read assigned monk skin tone value
    try:
        f=open(f"{abs_dir}\\{person}\\monk_scale_value.json")
        v=round(float(json.loads(f.read())['value']))
        f.close()
    except:
        print(f"There was a problem reading skin-tone {person}")
        continue

    #create cropped dir in data
    images=os.listdir(f"{abs_dir}\\{person}\\front-facing\\cropped")
    for image in images:
        try:
            print(f"Copying {person} data to ..\\clustered data_2\\{v}\\{image}")
            shutil.copyfile(
                f"{abs_dir}\\{person}\\front-facing\\cropped\\{image}",
                f"C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\clustered data_2\\{v}\\{image}") 
        except Exception as e:
            print(f"There was a problem copying data for {person}")
            print(e)
            pass