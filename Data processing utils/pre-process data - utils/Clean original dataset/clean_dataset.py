import os,shutil
import cv2 as cv

DATASET_DIR="D:\\Users\\Public\\Documents\\raw_data_2\\1-285"
HEAD_ROTATIONS=['front-facing','left-facing','right-facing','up-facing','down-facing']

if __name__ =="__main__":

    images_dir=os.listdir(DATASET_DIR)
    for img_dir in images_dir:
        for head_rot in HEAD_ROTATIONS:
            if '.py' in img_dir:
                continue
            img_dir_contents=os.listdir(f"{DATASET_DIR}\\{img_dir}\\{head_rot}")

            #Check for number of files
            if len(img_dir_contents)!=15:
                print(f"Not 15 images in: \\{img_dir}\\{head_rot}")

            #Check if mp4 exists, then delete it
            for file in img_dir_contents:
                if 'mp4' in file:
                    os.remove(f"{DATASET_DIR}\\{img_dir}\\{head_rot}\\{file}")
                    print(f"File deleted in: \\{img_dir}\\{head_rot}\\{file}")

            

