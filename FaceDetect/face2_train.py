import cv2 as cv
import os
from PIL import Image
import numpy as np

"""
程序说明：
    程序读入图像，检测得到图像中的人脸坐标并保存到列表中，同时另一个列表同一序号保存人脸的标号
    加载训练器，将人脸样本数组和对应标号数组送入训练器，最后保存模型数据供人脸检测匹配使用
"""

def getImageAndLabels(path):
    faceSamples = []
    ids = []
    name_paths = []

    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path,dir)) and len(os.listdir(os.path.join(path,dir))) != 0:
            name_paths.append(os.path.join(path, dir)) 
        else:
            continue
    print(name_paths) 
   
    id = 1
    for name_path in name_paths:
        imagePaths = [os.path.join(name_path,f) for f in os.listdir(name_path)]
        face_detector = cv.CascadeClassifier("./classifiers/haarcascade_frontalface_alt2.xml")
        
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            faces = face_detector.detectMultiScale(img_numpy)
            #id = str(os.path.split(imagePath)[1].split('.')[0])
            
            for x,y,w,h in faces:
                ids.append(id)
                if id == 4:
                    print("successfully detect liang")
                faceSamples.append(img_numpy[y:y+h,x:x+w])
        id+=1

    print("id:",id)

    return faceSamples,ids

if __name__ == '__main__':
    path = './faces/train'

    faces, ids = getImageAndLabels(path)

    recognizer = cv.face.LBPHFaceRecognizer_create()

    recognizer.train(faces, np.array(ids))

    recognizer.write('./trainer.yml')