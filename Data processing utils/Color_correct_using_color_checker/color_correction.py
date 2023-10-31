import cv2,sys 
import imutils
import numpy as np

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
img = cv2.imread("C:\\Users\\22218521\\Desktop\\Katlego Mbatha\\Collected data (2022)\\raw data\\1-126\\3\\front-facing\\20221007_102056.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,50,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
print("Number of contours detected:", len(contours))

for cnt in contours:
   x1,y1 = cnt[0][0]
   approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
   if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(cnt)
      ratio = float(w)/h

      print(f"width: {w}, height: {h}")

      if ratio >= 0.9 and ratio <= 1.1:
         img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
         cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      else:
         cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
         img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()