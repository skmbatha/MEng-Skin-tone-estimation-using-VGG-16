import os
import json
import math
import matplotlib.pyplot as plt

DATASET_DIR="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\raw data\\1-285"

dirs=os.listdir(DATASET_DIR)
output=[0]*10
x_axis=['1','2','3','4','5','6','7','8','9','10']

for dir in dirs:
    try:
        f=open(f"{DATASET_DIR}\\{dir}\\monk_scale_value.json",'r')
        data=f.read()
        value=json.loads(data)
        try:
            monk=round(float(value["value"]))
            output[int(monk)-1]+=1
        except:
            print("No value at {}".format(dir))
    except Exception as e:
        print(e)
        pass

# List of colors (for graph)
colors=['#f6ede4','#f3e7db','#f7ead0','#eadaba','#d7bd96','#a07e56','#825c43','#604134','#3a312a','#292420']
labels=[i for i in output] 

fig = plt.figure(figsize = (10, 5))
plt.bar(x_axis, output, color =colors,width = 1,label=labels)
plt.xlabel("Google monk scale value")
plt.ylabel("Number of identities")
plt.title("Distribution over {} identities.".format(len(dirs)))
plt.show()

print(output)
