import cv2
import os


# 加载训练数据集文件
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer.yml')
# 获取人脸模板名单
names = [ ]
name_path = './faces/train'
for dir in os.listdir(name_path):
    if os.path.isdir(os.path.join(name_path,dir)) and len(os.listdir(os.path.join(name_path,dir))) != 0:
        names.append(dir)
print(names)


def face_detect_demo(img):
    # 转换为灰度
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 加载人脸检测模型
    face_detector=cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_alt2.xml')
    # 设置模型参数
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))

    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        # 人脸匹配,评分低于50匹配较好，高于80很差
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 70:
            cv2.putText(img, names[ids-1], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)            
        else:
            # do something ...
            cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    cv2.imshow('result', img)


cap=cv2.VideoCapture(0)
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cap.release()
cv2.destroyAllWindows()

