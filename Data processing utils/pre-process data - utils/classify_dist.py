import os
import json
import math
import matplotlib.pyplot as plt


dirs=os.listdir(".")
output=[0]*10
x_axis=[1,2,3,4,5,6,7,8,9,10]

for dir in dirs:
    try:
        f=open(dir+'/monk_scale_value.json','r')
        data=f.read()
        value=json.loads(data)

        try:
            monk=round(float(value["value"]))
            output[int(monk)-1]+=1
        except:
            print("No value at {}".format(dir))
            
    except:
        pass

# List of colors (for graph)
colors=['#f6ede4','#f3e7db','#f7ead0','#eadaba','#d7bd96','#a07e56','#825c43','#604134','#3a312a','#292420']
labels=[i for i in output] 

fig = plt.figure(figsize = (10, 5))
plt.bar(x_axis, output, color =colors,width = 1,label=labels)
plt.xlabel("Google monk scale value")
plt.ylabel("Number of paricipats")
plt.title("Distribution over {} participants.".format(len(dirs)))
plt.show()

print(output)
