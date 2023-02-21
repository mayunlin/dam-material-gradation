import cv2
import numpy as np
img = cv2.imread('/home/chase/Boyka/stone/mmdetection/process/JP-1-01.jpg')
# img = cv2.resize(img, None, fx=0.25, fy=0.25)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=30, minRadius=100, maxRadius=500)
circles = np.uint16(np.around(circles))
print(circles)
for idx in circles[0, :]:
    cv2.circle(img, (int(idx[0]), int(idx[1])), int(idx[2]), (0, 255, 0), 2)  # draw the circle
    cv2.circle(img, (int(idx[0]), int(idx[1])), 2, (0, 0, 255), 3)  # draw the center of circle
cv2.imwrite('frame.jpg', img)
