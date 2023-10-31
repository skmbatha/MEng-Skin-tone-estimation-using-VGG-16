import sys
import os,shutil
import cv2 as cv

DATASET_DIR="D:\\Users\\Public\\Documents\\raw_data_2_train\\1-285"
HEAD_ROTATIONS=['front-facing','left-facing','right-facing','up-facing','down-facing']
IMAGE_NAMES=['F_warm_200','F_warm_600','F_warm_1200',
             'F_cool_200','F_cool_600','F_cool_1200',
             'L_warm_200','L_warm_600','L_warm_1200',
             'L_cool_200','L_cool_600','L_cool_1200',
             'H_warm_200','H_warm_600','H_warm_1200']

if __name__ =="__main__":

    images_dir=os.listdir(DATASET_DIR)
    for img_dir in images_dir:
        for head_rot in HEAD_ROTATIONS:
            if '.py' in img_dir:
                continue
            img_dir_contents=os.listdir(f"{DATASET_DIR}\\{img_dir}\\{head_rot}")

            #Rename_images
            counter=0
            sorted_img_dir_content=sorted(img_dir_contents)
            for image in sorted_img_dir_content:
                    os.rename(f"{DATASET_DIR}\\{img_dir}\\{head_rot}\\{image}",f"{DATASET_DIR}\\{img_dir}\\{head_rot}\\{IMAGE_NAMES[counter]}.jpeg")
                    counter+=1
            
    print("DONE!")
