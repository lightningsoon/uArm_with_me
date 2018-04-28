import os
import cv2
import glob
from random import shuffle
import numpy as np
from utils import coordinate_transformation as ct
dirname =  '../datasets/img/'

# (960,1280)numpy
size=(640,480) # size = (img_width, img_height)  # 宽 高
Psize=(600,400)#透视后的大小
def attain_img():
    Nub_cam = int(input('哪个摄像头0，1……'))
    assert Nub_cam >= 0
    # 和../ObjectDetection/train_my_ssd7.py中的size一样
    print("Ese退出，空格保存")
    N=len(glob.glob(dirname+'*.png'))
    cap = cv2.VideoCapture(Nub_cam)
    cv2.namedWindow('aa',cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('haha', size[0], size[1])

    while (True):
        ret, frame = cap.read()
        frame=cv2.resize(frame, size)
        cv2.imshow('aa',frame)
        flag = cv2.waitKey(60)
        if (flag == 32):
            cv2.imwrite(os.path.join(dirname, str(N) + 'xxx.png'), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
            N += 1
            print(N)
        elif (flag == 27):
            break


def rename_and_resize():
    imgs = glob.glob(dirname + '**.png')
    # imgs =os.listdir(dirname)
    # for i, each in enumerate(imgs):
    #     pic = cv2.imread(each)
        # pic = cv2.resize(pic, size)
        # cv2.imwrite(each, pic)
    # 乱序
    # shuffle(imgs)
    for i, each in enumerate(imgs):
        print(each,i)
        os.rename(each, dirname + "%04d.png" % i)
    print('完成格式化!')


def move():
    import shutil
    shutil.move('./label/train.csv', '../datasets/train.csv')
    shutil.move('./label/val.csv', '../datasets/val.csv')
    print("完成!")

def tran_color2hsv():
    while True:
        bgr=input('输入rgb用,分开,e退出')
        if bgr=='e':
            break
        bgr=list(map(int,bgr.split(',')[:]))
        print(cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV))

    pass

if __name__ == '__main__':
    print('你的图像存在这里', dirname)
    option = {"获取图像": attain_img, "重命名和尺寸": rename_and_resize, "搬动csv文件": move,"找出hsv":tran_color2hsv}
    choice = input("选择功能" + str(list(option.keys())) + str(list(range(len(option)))) + ":")
    choice = int(choice)
    print('你的选择是：', choice)
    if 0 <= choice < len(option):
        list(option.values())[choice]()
    else:
        print("没有这个选项，退出")
        exit(1)
