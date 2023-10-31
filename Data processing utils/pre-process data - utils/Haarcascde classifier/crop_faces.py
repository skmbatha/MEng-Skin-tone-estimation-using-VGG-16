"""
Read more about haar cascades @ Open CV
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""

import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

count=0
os.system("cls")
abs_dir="C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\1-290"
all_people=os.listdir(abs_dir)

for person in all_people:
    #create cropped dir in data
    try:
        os.mkdir(f"{abs_dir}\\{person}\\front-facing\\cropped")
    except:
        pass

    #Get front facing image list
    person_dir = f"{abs_dir}\\{person}\\front-facing"   
    cropped_dir= f"{abs_dir}\\{person}\\front-facing\\cropped" 

    images=os.listdir(person_dir)
    print(f"              -> {len(images)} images found")
    
    try:
        #Go through each image until a Face is found(min_width>=300px)
        pad=50
        face_found=False
        x = y = w = h = 0
        min_crop_width=300
        for img_label in images:
            if img_label=='cropped':
                continue
            try:
                img = cv2.imread(f"{person_dir}\\{img_label}")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.5, 5)
                print(f"              -> {len(faces)} faces found")            
                for (x_i, y_i, w_i, h_i) in faces:
                    x=x_i
                    y=y_i
                    w=w_i
                    h=h_i
                    if(w>min_crop_width): 
                        face_found=True
                        break
                if (face_found):
                    print(f"              -> {x} {y} {w} {h}")
                    break
            except:
                pass

        #Crop all image and save them
        for img_label in images:
            if img_label=='cropped':
                continue
            try:
                img = cv2.imread(f"{person_dir}\\{img_label}")
                img=img[y-pad:y+h+(pad*2),x-pad:x+w+(pad*2)]
                img=cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                cv2.imwrite(f"{cropped_dir}\\{img_label}", img)
            except:
                pass
            
    except Exception as e:
        print(e)
        pass
    print(f'Images done {count} {person}')
    count+=1

