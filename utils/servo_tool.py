import time

import serial


def function(demo):
    ser.write(demo)
    while True:
        time.sleep(0.2)
        if ser.readable():
            s = ser.readline()
            # print(s)
            if s==b'1\r\n':
                break

    # print(ser.readline())
    pass

ser = serial.Serial()
ser.baudrate = 9600
ser.port = '/dev/cu.usbmodem1411'
print(ser)
ser.open()
print(ser.is_open)

while (1):

    # demo = input("press any key ")
    # demo = b"90,70,70"
    # for x in range(60,120,2):
    #     demo = b'%02d,100,70' % (x)
    #     print(x)
    #     function(demo)
    #     pass
    for x in range(60,140,4):
        demo = b'90,%02d,30' % (x)
        print(x)
        function(demo)
        pass
    for x in range(10,60,4):
        demo = b'90,90,%02d' % x
        print(x)
        function(demo)
        pass
    # 默认"90(-底座顺时针),90(-大臂向下),60(+小臂向下),90,90"



