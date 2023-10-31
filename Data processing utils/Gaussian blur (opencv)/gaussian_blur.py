import cv2 as cv
import numpy as np

img=cv.imread("data/1.jpg")
# apply guassian blur on src image
for i in range(0,20):
    img = cv.GaussianBlur(img,(223,223),cv.BORDER_DEFAULT)
# display input and output image
cv.imshow("Gaussian Smoothing",img)
cv.waitKey(0) # waits until a key is pressed
cv.destroyAllWindows() # destroys the window showing image