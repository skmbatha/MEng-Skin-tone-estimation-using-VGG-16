import pandas as pd
import numpy as np
import os,sys
import matplotlib.pyplot as plt

COLORS=["#f6ede4","#f3e7db","#f7ead0","#eadaba","#d7bd96","#a07e56","#825c43","#604134","#3a312a","#292420"]
INPUT_FILE="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\CNN_training_models\\VGG16-PyTorch_Regression - Lab\\validation_output.csv"
df=pd.read_csv(INPUT_FILE)

DATA={}
for index in range(1,11):
    DATA[index]=[]

for col in df.iterrows():
    DATA[round(col[1][2])].append(col[1][3])

OUTPUT=[]
for index in range(1,11):
    OUTPUT.append(DATA[index])

#print categorised data
l1 = np.array(OUTPUT[0]+OUTPUT[1]+OUTPUT[2])
l2 = np.array(OUTPUT[3]+OUTPUT[4]+OUTPUT[5]+OUTPUT[6])
l3 = np.array(OUTPUT[7]+OUTPUT[8]+OUTPUT[9])


print(f"H1: MEAN: {np.mean(l1)} ; STD : {np.std(l1)} - {len(l1)}")
print(f"H2: MEAN: {np.mean(l2)} ; STD : {np.std(l2)} - {len(l2)}")
print(f"H3: MEAN: {np.mean(l3)} ; STD : {np.std(l3)} - {len(l3)}")
sys.exit()

BAR=[]
for row in OUTPUT:
    BAR.append(len(row))

#DRAW BOX AND WISKAR PLOT
fig1,ax1=plt.subplots()
bp=ax1.boxplot(OUTPUT,showfliers=False,patch_artist=True,notch=True)

i=0
for box in bp["boxes"]:
    box.set_facecolor(COLORS[i])
    i+=1

ax1.set_title("L2 distance error vs skin-tone")
plt.xlabel("Monk skin-tone")
plt.ylabel("L2 distance error")
plt.show()


#DRAW DATASET SKIN DISTRIBUTION
fig1,ax1=plt.subplots()
plt.bar([1,2,3,4,5,6,7,8,9,10], BAR, color =COLORS, width = 0.4)
ax1.set_title("Skin-tone distribution")
plt.xlabel("Monk skin-tone")
plt.ylabel("Number of images")
plt.show()