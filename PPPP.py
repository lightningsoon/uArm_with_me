
import math
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
img = cv2.imread('1.png')
# ->HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 阈值
lower_green = np.array([30, 100, 130])
upper_green = np.array([45, 210, 255])
#
mask = cv2.inRange(hsv_image, lower_green, upper_green)
hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
# hsv->gray
img = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
y,x=img.shape
blur = cv2.GaussianBlur(img, (5, 5), 0)

corners = cv2.goodFeaturesToTrack(blur,25,0.01,10) # 返回的结果是 [[ 311., 250.]] 两层括号的数组。 corners = np.int0(corners)
# (x,1,2)
corners=np.array(corners[:,0,:])

cc=[]
for p in [[0,y],[x,y]]:
    d = np.linalg.norm(corners - np.array([p]), axis=1)
    c = corners[np.argmin(d)]
    if math.tan(math.pi/3)>(abs(c[1]-p[1]))/abs(c[0]-p[0])>math.tan(math.pi/18):
        cc.append(c)
if len(cc)==2:
    c=cc[random.randint(0,1)]
else:
    c=cc[0]
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
cv2.circle(img,tuple(c),3,(0,0,255))

cv2.imshow('ff',img)
cv2.waitKey(0)