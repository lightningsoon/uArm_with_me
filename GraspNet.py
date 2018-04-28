'''
搭建一个神经网络
输入图像
输出分类及抓取位置
'''
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
import cv2
import numpy as np


from utils.coordinate_transformation import fun
from ObjectDetection.keras_layer_AnchorBoxes import AnchorBoxes
from ObjectDetection.keras_ssd_loss import SSDLoss
from ObjectDetection.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2

from ObjectDetection.train_my_ssd7 import classes
from utils.image_process import size,Psize
# from main import destroyCVWindow

def destroyCVWindow():
    # plt.close('all')
    for i in range(4):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def start_detecting():

    global net_control,ROI,graspPoint,coor0
    # 打开视频
    print("success")
    cap=cv2.VideoCapture(0)
    # print('???')
    cv2.namedWindow('haha', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('haha', size[0], size[1])
    print("start")
    # M=np.load('./MMM.npy')
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, size)
        frame, coor = fun(frame)
        # print(coor)

        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow('haha',frame)
        flag=cv2.waitKey(30)&0xFF
        if flag == 32:
            print('完成自动标定，下一步请按空格，重新标定按任意键')
            k = cv2.waitKey(0) & 0xFF
            if k == 32:
                coor0.put(coor)
                print(coor)
                print('退出标定')
                destroyCVWindow()
                net_control.put('put')
                stop_flag = True
                break
            else:
                continue
        elif flag == 27:
            print('退出')
            destroyCVWindow()
            exit(0)

    # 载入模型
    K.clear_session()
    model_path = './ObjectDetection/ssd7.h5'
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    K.clear_session()
    print("放杯子")
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'compute_loss': ssd_loss.compute_loss})
    # 读取参数
    model.load_weights('./ObjectDetection/ssd7_weights.h5')
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
    # model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    while True:
        ret,frame=cap.read()
        frame=cv2.resize(frame, size)

        # frame = cv2.warpPerspective(frame, M, Psize)
        frame0=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame0=np.expand_dims(frame0,axis=0)
        ## 进行检测
        y_pred=model.predict(frame0)
        y_pred_decoded = decode_y2(y_pred,
                                   confidence_thresh=0.6,
                                   iou_threshold=0.5,
                                   # top_k='all',
                                   top_k=1,
                                   input_coords='centroids',
                                   normalize_coords=False,
                                   img_height=None,
                                   img_width=None)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        # print("Decoded predictions (output format is [class_id, confidence, xmin, ymin, xmax, ymax]):\n")
        # print('y',y_pred_decoded[0])

        for box in y_pred_decoded[0]:
            xmin,ymin,xmax,ymax = map(int,[box[-4],box[-3],box[-2],box[-1]])
            # print(xmin,ymin)
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            # 画框
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color=(0,0,255))
            cv2.putText(frame,label,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, 8)
        # TODO 显示抓取点
        # if not graspPoint.empty():
        #     GraspP = graspPoint.get()
        #     frame0=np.copy(frame)
        #     cv2.circle(frame0, tuple(GraspP), 2, (0, 0, 255), -1)
        #     cv2.imshow('graspPoint',frame0)

        # if (len(y_pred_decoded[0])!=0) and ROI.empty():
        #     print('roi')
        #     ROI.put_nowait([frame[ymin:ymax, xmin:xmax,:], (xmin, ymin), (xmax, ymax)])
        if (len(y_pred_decoded[0])!=0) and not net_control.empty() :
            print('nc', net_control.empty())
            n_c_g=net_control.get()
            print(n_c_g)
            if (n_c_g=='put'):
                # 检测到了放进盒子里
                    if ROI.empty():
                        print('roi')
                        ROI.put_nowait([frame[ymin:ymax, xmin:xmax,:],(xmin,ymin),(xmax,ymax)])
                #     y_pred_decoded[0]:
                #     [   1.  ,    0.65,  222.39,    7.61,  345.27,  190.19],
                #     [   1.  ,    0.65,  152.86,  197.05,  313.62,  296.81],
                #     [   1.  ,    0.66,  103.21,  142.65,  209.86,  291.63]
            elif n_c_g=='exit':
                break
        cv2.imshow('ooo', frame)
        k = cv2.waitKey(40) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
    pass

if __name__ == '__main__':
    from multiprocessing import Process
    from multiprocessing import Queue
    from time import sleep

    import Motion

    def start(p):
        for i in range(len(p)):
            print('\n开始准备_{0}'.format(p[i].name))
            p[i].daemon = True
            p[i].start()

    coor0=Queue(1)
    ROI = Queue(1)  # 检测框
    net_control = Queue(1)  # 控制神经网络是否开始检测
    graspPoint = Queue(1)  # 抓取点
    print('启动厨房餐具回收机器人……')

    p = []
    p.append(Process(target=Motion.move, args=(ROI, net_control, graspPoint,coor0), name='机械臂'))
    start(p)
    start_detecting()
    # p.append(Process(target=start_detecting, args=(net_control, ROI, graspPoint), name='神经网络'))

    sleep(1)
    print('正在启动!')



    # 同时启动3个进程
    # while True:
    #     if coor0 !=None:
    #         break

    while True:
        E = input("输入e退出")
        if E == 'e':
            break
    net_control.put_nowait(E)
    print("感谢使用！")