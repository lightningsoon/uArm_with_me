# from multiprocessing import Process
# from multiprocessing import Queue
# from time import sleep
# import cv2
# from threading import Thread
#
# from matplotlib import pyplot as plt
# plt.ion()
#
# from GraspNet import start_detecting
# from Motion import move
# from utils.coordinate_transformation import fun
#
# stop_flag=None
# def destroyCVWindow():
#     plt.close('all')
#     for i in range(4):
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#
# def pre_pro():
#     global stop_flag,flag
#     from utils.image_process import size
#     cap = cv2.VideoCapture(0)
#     cv2.namedWindow('haha', cv2.WINDOW_AUTOSIZE)
#     cv2.resizeWindow('haha', size[0], size[1])
#     while True:
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, size)
#         frame, coor = fun(frame)
#         # print(coor)
#
#         plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         # cv2.imshow('haha',frame)
#
#         if flag == 32:
#             print('完成自动标定，下一步请按空格，重新标定按任意键')
#             k = cv2.waitKey(0) & 0xFF
#             if k == 32:
#                 print('退出标定')
#                 destroyCVWindow()
#                 cap.release()
#                 stop_flag=True
#                 break
#             else:
#                 continue
#         elif flag == 27:
#             print('退出')
#             destroyCVWindow()
#             cap.release()
#             exit(0)
#         elif flag == ord('l'):
#             stop_flag=True
#             destroyCVWindow()
#             cv2.waitKey(1)
#
#
#
#
# def start(p):
#     for i in range(len(p)):
#         print('\n开始准备_{0}'.format(p[i].name))
#         p[i].daemon=True
#         p[i].start()
#
# if __name__ == '__main__':
#
#     ROI=Queue(1)#检测框
#     net_control=Queue(1)#控制神经网络是否开始检测
#     graspPoint=Queue(1)#抓取点
#     print('启动厨房餐具回收机器人……')
#
#     p=[]
#     p.append(Process(target=start_detecting,args=(net_control,ROI,graspPoint),name='神经网络'))
#     p.append(Process(target=move, args=(ROI,net_control,graspPoint), name='机械臂'))
#
#     sleep(2)
#     print('正在启动!')
#
#     start(p)
#
#     # 同时启动3个进程
#
#
#
#
#
#     net_control.put('put')
#     while True:
#         E=input("输入exit退出")
#         if E=='exit':
#             break
#     net_control.put_nowait(E)
#     print("感谢使用！")