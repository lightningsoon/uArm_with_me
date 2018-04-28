'''
与下位机通信
发送接收数据
'''
from time import sleep
import cv2
import serial
import numpy as np
import math
import random
from sklearn.externals import joblib
from sklearn import preprocessing

# 阈值
lower_green = np.array([30, 100, 130],dtype=np.uint8)
upper_green = np.array([45, 210, 255],dtype=np.uint8)
#


square_x,square_y=(300,210)

poly=preprocessing.PolynomialFeatures(degree=2)
model=joblib.load('./utils/model.m')
# print(model)

def PPP(roi):
    # 提取抓取点
    img,(xmin,ymin),(xmax,ymax)=roi
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    # hsv->gray
    img = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    y0, x0 = img.shape
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    corners = cv2.goodFeaturesToTrack(blur, 25, 0.01, 10)  # 返回的结果是 [[ 311., 250.]] 两层括号的数组。 corners = np.int0(corners)
    # (x,1,2)
    corners = np.array(corners[:, 0, :])

    cc = []
    for p in [[0, y0], [x0, y0]]:
        d = np.linalg.norm(corners - np.array([p]), axis=1)
        c = corners[np.argmin(d)]
        if math.tan(math.pi / 3) > (abs(c[1] - p[1])) / abs(c[0] - p[0]) > math.tan(math.pi / 39):
            cc.append(c)
    if len(cc) == 2:
        point = cc[random.randint(0, 1)]
    elif len(cc)==1:
        point = cc[0]
    else:
        point=((xmax-xmin)*7/10,ymax-ymin-20)
    # cv2.circle(img, tuple(point), 2, (0, 0, 255))


    return point[0]+xmin,point[1]+ymin

def Tran(point,coor):

    # 将坐标点转换成角度
    # 相机放在以机械臂这侧的右下角
    angle=[0]*3
    left, up, right, down = coor
    x, y = square_x * ((point[1] - left[1]) / (down[1] - left[1]) - 1 / 2), square_y * (point[0] - down[0]) / (
                right[0] - down[0])
    # 注意距离，上面是错的

    # 预测后两个

    #与原点距离

    distance=np.linalg.norm([x,y])
    distance=poly.fit_transform([[distance]])
    # print(model.predict(distance))
    angle[1],angle[2]=model.predict(distance)[0]
    # 计算第一个
    angle[2]-=3
    angle[1]+=2

    angle[0]=90+math.atan2(x,y)
    # 公式的原点上面写错了
    if angle[0]<89:

        angle[0]-=17
    elif angle[0]>98:
        angle[0]+=3
    print('angle',angle)
    return angle#xx,xx,xx

def move(ROI,net_control,graspPoint,coor0):

    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port = '/dev/cu.usbmodem1411'
    # print(ser)
    ser.open()
    sleep(2)
    print('串口打开',ser.is_open)
    while True:
        # sleep(0.5)
        if not ROI.empty():
            print('???')
            point=PPP(ROI.get_nowait())
            if not coor0.empty():
                coooo=coor0.get_nowait()
            tran=Tran(point,coooo)
            if tran!=None:
                print(tran,type(tran))
                ser.write(b'%d,%d,%d' % (tran[0],tran[1],tran[2]))
                tran=None
            # 写完了，等待返回
            while True:
                s = ser.read(1)
                print(s)
                if s=='0' and not net_control.full():
                    graspPoint.put(point)
                    net_control.put('show')
                elif s==b'1' and not net_control.full():
                    net_control.put('put')
                    break
                    pass
        else:
            continue



    pass




