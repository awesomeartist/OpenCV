"""
程序说明：程序通过调用连接到电脑的摄像头采集视频，在采集过程中等待按键输入's'获得当前视频帧并保存到设定的文件夹中，按下空格结束图像采集
"""

import cv2 as cv
#调用摄像头0采集视频
cap = cv.VideoCapture(0)
#num用于标记图像采集的数量，当前保存的是第几张图像
num = 1
#采集对象的名称
name = "liangxinmind"

while(cap.isOpened()):
    flag, Vshow = cap.read()
    cv.imshow("Capture_Test", Vshow)
    k = cv.waitKey(1)&0xff
    if k == ord('s'):
        cv.imwrite("./faces/train/Liang/"+str(num)+name+".jpg", Vshow)
        print("success to save "+str(num)+name+".jpg")
        print("-----------------")
        num+=1
    elif k == ord(' '):
        break

cap.release()
cv.destroyAllWindows()