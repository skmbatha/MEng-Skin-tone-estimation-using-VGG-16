"""
Read more about haar cascades @ Open CV
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""

import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_crop_face(images :list,pad:int =50, dimension:int = 200) -> dict:    
    try:
        #Go through each image until a Face is found(min_width>=300px)
        face_found=False
        x = y = w = h = 0
        output_images=[]
        
        # Find image in images
        for img_dir in images:
            if img_dir=='cropped':
                continue
            try:
                img = cv2.imread(img_dir)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.5, 5)
                #print(f"              -> {len(faces)} faces found")            
                for (x_i, y_i, w_i, h_i) in faces:
                    x=x_i
                    y=y_i
                    w=w_i
                    h=h_i
                    if(w>dimension): 
                        face_found=True
                        break
                if (face_found):
                    #print(f"              -> {x} {y} {w} {h}")
                    break
            except:
                pass

        # Crop all image and save them
        for img_dir in images:
            if img_dir=='cropped':
                continue
            try:
                img = cv2.imread(img_dir)
                img=img[y-pad:y+h+(pad*2),x-pad:x+w+(pad*2)]
                img=cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                output_images.append({'name':str(img_dir).split("\\")[-1],'image':img})
            except:
                pass

        # sReturn the data back
        return output_images
            
    except Exception as e:
        print(f"Something went wrong: {e}")
        pass


