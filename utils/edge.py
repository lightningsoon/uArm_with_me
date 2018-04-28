import cv2
import numpy as np
import math

frame = cv2.imread("1.png")
#cv2.imshow("frame", frame)

h, w = frame.shape[:2]

blured = cv2.blur(frame, (5, 5))

hsv=cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
lower_green = np.array([30, 100, 160])
upper_green = np.array([45, 210, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
hsv_image = cv2.bitwise_and(hsv, hsv, mask = mask)
# cv2.imshow("mask", mask)

gray = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(gray, 0, 50)
#cv2.imshow("edges", edges)

ret, imag = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("threshold", imag)

_,contours, hierarchy = cv2.findContours(imag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
n=0

for cnt in contours:
    if len(cnt) > 50:
        try:
            ellipse = cv2.fitEllipse(cnt)
            S1 = cv2.contourArea(cnt)
            S2 = math.pi * ellipse[1][0] * ellipse[1][1]
        except cv2.error:
            continue

        if (S1 / S2) < 0.4:
            cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
            cv2.imshow('', frame)
            n += 1
            print(n)
cv2.waitKey(0) & 0xFF

exit(0)