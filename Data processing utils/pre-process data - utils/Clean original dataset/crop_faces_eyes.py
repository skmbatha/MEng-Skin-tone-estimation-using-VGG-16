"""
Read more about haar cascades @ Open CV
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""

import cv2
import os
import sys

DATASET_DIR="D:\\Users\\Public\\Documents\\raw_data_2_train\\1-285"

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

def find_head_from_eyes(imgs_dir):
        
        pad_x=10
        sqr_tol=15
        img_list=os.listdir(imgs_dir)
        
        #Try over 3 attempts
        for iter in range(1,4):

            #Iterate thorugh images
            for sqr_dist in range(50,0,-10):
                for img_name in img_list:
                    img = cv2.imread(f"{imgs_dir}\\{img_name}")
                    img = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    eyes = list(eyes_cascade.detectMultiScale(gray, 1.5, 1))
                    #print(eyes)
                    
                    #try to fix if > 2 eyes
                    if len(eyes) > 2 and iter==2:
                        for i in range(1,len(eyes)):
                            if abs(eyes[0][0]-eyes[i][0])>=sqr_dist:
                                eyes[1]=eyes[i]
                                break
                        eyes=eyes[:1]

                    #estimate left position if one size is found
                    if len(eyes) == 1 and iter==3:
                        #if estimated left eye
                        if eyes[0][0]<500:
                            (x_i, y_i, w_i, h_i) =eyes[0]
                            eyes.append(((x_i+w_i+35),y_i,w_i,h_i))

                        #if estimated right eye
                        if eyes[0][0]>500:
                            (x_i, y_i, w_i, h_i) =eyes[0]
                            eyes.append(((x_i-w_i-35),y_i,w_i,h_i))

                    #Check if eyes are found
                    if len(eyes)==2:
                        if (eyes[1][0]<eyes[0][0]):
                            temp=eyes[1]
                            eyes[1]=eyes[0]
                            eyes[0]=temp

                        (x1_i, y1_i, w1_i, h1_i) =eyes[0]
                        (x2_i, y2_i, w2_i, h2_i) =eyes[1]

                        #Check square size compare
                        if (abs(w1_i-w2_i)<=sqr_tol and abs(h1_i-h2_i)<=sqr_tol):
                            #Check squares' distance from each other
                            if ((x2_i-x1_i)>=sqr_dist):
                                s_w = 2*pad_x + x2_i+w2_i -x1_i
                                s_h = s_w

                                p_x=x1_i-pad_x
                                p_y=round(y1_i-((s_w/2)-(h2_i/2))/2)
                                
                                return (p_x,p_y,s_w,s_h)
                            
        return None

def find_head_from_face(imgs_dir):

    images=os.listdir(imgs_dir)
    
    try:
        pad=40
        face_found=False
        x = y = w = h = 0
        min_crop_width=70
        for img_label in images:
            if img_label=='cropped':
                continue
            try:
                img = cv2.imread(f"{imgs_dir}\\{img_label}")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.5, 5)          
                for (x_i, y_i, w_i, h_i) in faces:
                    x=x_i
                    y=y_i
                    w=w_i
                    h=h_i
                    if(w>min_crop_width): 
                        face_found=True
                        break
                if (face_found):
                    return (x+pad,y+pad,w-pad*2,h-pad*2)
            except Exception as e:
                print(e)
                pass
    except Exception as e:
        print(e)
    return None

if __name__ == "__main__":

    identities_dir=os.listdir(DATASET_DIR)
    for identity in identities_dir:

        head=find_head_from_eyes(f"{DATASET_DIR}\\{identity}\\front-facing")
        if head == None:
            head=find_head_from_face(f"{DATASET_DIR}\\{identity}\\front-facing")

        if head is not None:
            images=os.listdir(f"{DATASET_DIR}\\{identity}\\front-facing")
            for image in images:
                img = cv2.imread(f"{DATASET_DIR}\\{identity}\\front-facing\\{image}")

                (x_i, y_i, w_i, h_i) = head 
                img=img[y_i:y_i+h_i,x_i:x_i+w_i]
                img = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA) 
                               
                """cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()"""

                cv2.imwrite(f"{DATASET_DIR}\\{identity}\\front-facing\\{image}", img)
