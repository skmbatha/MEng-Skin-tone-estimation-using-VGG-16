
import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0
os.system("cls")
abs_dir="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\clustered data_2\\brightness_2_3_aug\\validation" #change to 'training' / 'validation'
all_classes=os.listdir(abs_dir)

for skin_tone in all_classes:
    images=os.listdir(f"{abs_dir}\\{skin_tone}")
    for image in images:
        img = cv2.imread(f"{abs_dir}\\{skin_tone}\\{image}")
        img = cv2.flip(img, 1)
        cv2.imwrite(f"{abs_dir}\\{skin_tone}\\flipped_{image}", img)
        print(f'Flipping images in {skin_tone}\\{image}')

