import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib

# 加载训练数据集文件
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
names=[]
warning_times = 0

# 对字符串进行哈希编码
def md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

# 短信发送状态，字典数据类型
statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持,请确认支持curl或者FSocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}

# 在检测到报警信号后，发送信息到相应手机
def warning():
    sms_api = "http://api.smsbao.com/"
    # 短信平台账号
    user = '13******10'
    # 短信平台密码
    password = md5('*******')
    # 要发送的短信内容
    content = '【报警】\n原因：检测到未知人员\n地点：xxx'
    # 要发送短信的手机号码
    phone = '*******'

    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = sms_api + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])

# 准备识别的图片
def face_detect_demo(img):
    # 转换为灰度
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 加载人脸检测模型
    face_detector=cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_alt2.xml')
    # 设置模型参数
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))

    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        # 人脸匹配，评分越高越不可信，出现次数过多
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence > 80:
            warning_times += 1
            if warning_times > 100:
               warning()
               warning_times = 0
            cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result',img)
    #print('bug:',ids)

def name():
    path = './data/jm/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


cap=cv2.VideoCapture('1.mp4')
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
#print(names)
