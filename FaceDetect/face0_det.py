"""
程序说明：读入图像转化为灰度图，加载人脸检测模型，得到人脸矩形坐标并绘制矩形到原图像上
"""

import cv2 as cv

def face_detect():
    #读入图像并转化为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #加载人脸检测模型
    face_classifier = cv.CascadeClassifier("./classifiers/haarcascade_frontalface_alt2.xml")
    #设置参数并检测图像
    face = face_classifier.detectMultiScale(gray, 1.01, 5)
    #将检测到的人脸坐标读出并绘制矩形
    for x, y, w, h in face:
        cv.rectangle(img, (x,y), (x+w,y+h), color=(0,0,255), thickness= 2)
    
    cv.imshow("face", img)

img = cv.imread("./faces/train/Ben Afflek/1.jpg")


face_detect()
#等待按键'q'输入，退出窗口
while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()
