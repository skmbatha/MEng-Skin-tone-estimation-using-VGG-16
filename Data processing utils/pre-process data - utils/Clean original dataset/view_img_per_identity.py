import cv2
import os
import sys

DATASET_DIR="D:\\Users\\Public\\Documents\\raw_data_2_train\\1-285"


if __name__ == "__main__":

    identities_dir=os.listdir(DATASET_DIR)
    for identity in identities_dir:
        images=os.listdir(f"{DATASET_DIR}\\{identity}\\front-facing")
        img = cv2.imread(f"{DATASET_DIR}\\{identity}\\front-facing\\{images[3]}")
        cv2.imshow(f"Identity : {identity}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()