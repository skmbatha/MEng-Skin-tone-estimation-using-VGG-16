
import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0
os.system("cls")
abs_dir="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\pre-process data - utils\\create random-set\\input_data\\pad_50_3\\images" #change to 'training' / 'validation'
all_classes=os.listdir(abs_dir)

for skin_tone in all_classes:
    images=os.listdir(f"{abs_dir}\\{skin_tone}")
    for image in images:
        img = cv2.imread(f"{abs_dir}\\{skin_tone}\\{image}")
        
        y=0;x=0;pad=-35
        h, w, channels = img.shape
        img=img[y-pad:y+h+(pad*2),x-pad:x+w+(pad*2)]
        img=cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)

        """if os.path.exists(f"{abs_dir}\\{skin_tone}\\cropped") == False:
            os.mkdir(f"{abs_dir}\\{skin_tone}\\cropped")"""
        
        cv2.imwrite(f"{abs_dir}\\{skin_tone}\\cropped_{image}", img)
        print(f'Cropping images in {skin_tone}\\{image}')

        os.remove(f"{abs_dir}\\{skin_tone}\\{image}")

