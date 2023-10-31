#https://jmanansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886de

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle

img = "test_images\\20221007_102101.jpg"
dinner = imread(img)
plt.imshow(dinner, cmap='gray')

fig, ax = plt.subplots(1,2, figsize=(10,6))
#ax[0].imshow(dinner)
#ax[0].set_title('Original Image')
dinner_max = (dinner*1.0 / dinner.max(axis=(0,1)))
ax[1].imshow(dinner_max)
cv2.waitKey(0)
cv2.destroyAllWindows()
ax[1].set_title('Whitebalanced Image')