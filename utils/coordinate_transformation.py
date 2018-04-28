import cv2
import numpy as np



# 第一次运行时确定位置
def fun(frame):
    row_image=np.copy(frame)
    # ->HSV
    hsv_image = cv2.cvtColor(row_image, cv2.COLOR_BGR2HSV)
    # 阈值
    lower_green = np.array([32, 90, 100],dtype=np.uint8)
    upper_green = np.array([50, 210, 255],dtype=np.uint8)
    #
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    # hsv->gray
    img = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # %%
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    # th_ = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    # %%
    _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # %%
    up, down, left, right = map(np.array, [[1e4, 1e4], [0, 0], [1e4, 1e4], [0, 0]])
    renew = [False] * 4
    for c in contours:
        c = c[:, 0, :]
        for i in [np.argmin(c[:, 0])]:
            if c[i][0] < left[0]:
                left = c[i]
                renew[0] = True
        for i in [np.argmin(c[:, 1])]:
            if c[i][1] < up[1]:
                up = c[i]
                renew[1] = True
        for i in [np.argmax(c[:, 0])]:
            if c[i][0] > right[0]:
                right = c[i]
                renew[2] = True
        for i in [np.argmax(c[:, 1])]:
            if c[i][1] > down[1]:
                down = c[i]
                renew[3] = True
    if (renew[0] and renew[1] and renew[2] and renew[3]):
        # print(tuple(left),tuple(up),tuple(right),tuple(down))
        cv2.putText(row_image, str(left), tuple(left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(row_image, str(up), tuple(up), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(row_image, str(right), tuple(right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        cv2.putText(row_image, str(down), tuple(down), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.circle(row_image, tuple(left), 2, (0, 0, 255), -1)
        cv2.circle(row_image, tuple(up), 2, (0, 0, 255), -1)
        cv2.circle(row_image, tuple(right), 2, (0, 0, 255), -1)
        cv2.circle(row_image, tuple(down), 2, (0, 0, 255), -1)
        renew = [False] * 4
    # cv2.imshow('row_image', row_image)
    # 透视变换
    # pts1 = np.float32([up, right, down, left])
    # pts2 = np.float32([[200, 200], [200+Psize[0], 200], [Psize[0]+200,Psize[1]+200], [200, 200+Psize[1]]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # print(M)
    # tran_image = cv2.warpPerspective(frame, M, (Psize[0]+400,Psize[1]+400))
    return row_image,(tuple(left),tuple(up),tuple(right),tuple(down))

def main():
    global Psize,size
    cap = cv2.VideoCapture(0)
    # cv2.namedWindow('row_image',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('tran_show',cv2.WINDOW_NORMAL)
    # frame=cv2.imread('./IMG_3591.JPG')
    while True:
        ret, frame = cap.read()
        frame=cv2.resize(frame,size)
        tran_image=fun(frame)
        cv2.imshow('tran_show', tran_image)
        # cv2.imwrite('1.png',tran_image)

        k = cv2.waitKey(30) & 0xFF
        if (k == 27):
            break
        if (k == 32):
            # print('M保存成npy文件')
            # np.save('./MMM.npy', M)
            # print(M)
            #
            # M0=np.load('./MMM.npy')
            # print(M0)
            pass
    pass


if __name__ == '__main__':
    import utils.image_process as uip
    Psize, size = uip.Psize, uip.size
    main()
'''
604 336
300 151
27 222
286 271
'''
'''
(222, 27) (151, 44) (80, 92) (215, 30)
'''