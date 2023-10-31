import os,shutil
import cv2 as cv

DATASET_DIR="D:\\Users\\Public\\Documents\\raw_data_2_train\\1-285"
HEAD_ROTATIONS=['front-facing','left-facing','right-facing','up-facing','down-facing']

if __name__ =="__main__":

    images_dir=os.listdir(DATASET_DIR)
    for img_dir in images_dir:
        for head_rot in HEAD_ROTATIONS:
            if '.py' in img_dir:
                continue
            img_dir_contents=os.listdir(f"{DATASET_DIR}\\{img_dir}\\{head_rot}")

            #Scale down image
            for file in img_dir_contents:
                    image=cv.imread(f"{DATASET_DIR}\\{img_dir}\\{head_rot}\\{file}")
                    resized = cv.resize(image, (224,224), interpolation = cv.INTER_AREA)
                    cv.imwrite(f"{DATASET_DIR}\\{img_dir}\\{head_rot}\\{file}",resized)

            

