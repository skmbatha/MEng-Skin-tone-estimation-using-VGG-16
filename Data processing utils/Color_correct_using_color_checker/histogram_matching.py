import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import argparse

source = "test_images/20221012_160538.jpg" 
reference = "test_images/20221007_121054.jpg"

#load the source and reference images
print("[INFO] loading source and reference images...")
src = cv2.imread(source)
ref = cv2.imread(reference)

#reshape the image
src=cv2.resize(src, (224,224))
ref=cv2.resize(ref, (224,224))

# determine if we are performing multichannel histogram matching
# and then perform histogram matching itself
print("[INFO] performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref,channel_axis=-1)

# show the output images
cv2.imshow("output",np.concatenate((src,ref,matched),axis=1)) #for V1
#cv2.imshow("Source", src)
#cv2.imshow("Reference", ref)
#cv2.imshow("Matched", matched)
cv2.waitKey(0)