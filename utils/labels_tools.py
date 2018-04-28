'''
图片标注工具
'''

import cv2
import csv
import os
import glob
import numpy as np
import copy
# 当鼠标按下时变为 True
drawing = False
ix, iy = -1, -1


# 创建回调函数
def draw_rectangle(event, x, y, flags, param):
    global jx, jy
    global ix, iy, drawing, mode, row_image, image

    # 当按下左键是返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
    elif drawing == True and event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        image = np.copy(row_image)
        jx, jy = x, y
        cv2.rectangle(image, (ix, iy), (jx, jy), (0, 255, 0), 1)
        cv2.putText(image, "%d,%d" % (ix, iy), (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
        cv2.putText(image, "%d,%d" % (jx, jy), (jx, jy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
    # 当鼠标松开停止绘画。
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# 下面我们要把这个回调函数与 OpenCV 窗口绑定在一起。

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

header = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']  # 格式

def main(C, dirs):
    global row_image, image,row,classes,ix,iy,jx,jy
    ii=int(input("上次停的位置"))
    img_names = glob.glob(dirs + '*.png')
    number=len(img_names)
    num = {"train":int(0.8*number), "val":number-int(0.8*number)}
    sum=0
    abc=list(str(j) for j in range(1, len(classes)))
    def write_k_class(img_name,k):
        row = [os.path.basename(img_name), ix, jx, iy, jy, k]
        f_csv.writerow(row)
        print(row)

    for name in list(num.keys()):
        row=copy.deepcopy(header)
        with open('./label/'+name+'.csv', 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(row)  # 头必须要写@ssd_batch_generator.py line330
            for i in range(sum,sum+num[name]):
                # if i< ii-1:
                #     continue
                print()
                print((i+1)+sum)
                row_image = cv2.imread(img_names[i])
                image = np.copy(row_image)
                while (True):
                    try:
                        cv2.imshow('image', image)
                    except TypeError:
                        print(image)
                        break
                    k = cv2.waitKey(70) & 0xFF
                    if k == 27:
                        f.close()
                        break
                    elif k == 32:
                        row = [os.path.basename(img_names[i]), ix, jx, iy, jy, C]
                        f_csv.writerow(row)
                        print(img_names[i],row)
                        # 空格默认写csv文件
                        pass
                    elif chr(k) in abc:
                        write_k_class(img_names[i],chr(k))

                    elif k == ord('n'):
                        #跳过
                        break
                    else:
                        # 其他 键无效
                        continue
                ix,jx,iy,jy=0,0,0,0

                if k == 27:
                    break
        if k == 27:
            break
        sum = sum + num[name]

    cv2.destroyAllWindows()

    pass


if __name__ == '__main__':
    import utils.image_process as ip
    classes = ['backgroud', 'cup']
    C = input("输入你默认标记的组别...\n"+str(list(range(len(classes))))+'=='+str(classes)+':')
    assert os.path.isdir(ip.dirname)
    if 1<=int(C)<len(classes):
        print("按数字键标记不同类别，ESC退出，空格默认组别，回车下一张")
        main(C, ip.dirname)
    else:
        print('程序退出！')


'''

0075 1->2
0052 3->2
0068 3->2
'''