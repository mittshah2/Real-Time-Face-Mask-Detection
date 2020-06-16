from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np


model=load_model('mask_model.h5')

face_cascade=cv2.CascadeClassifier('face_cascade.xml')

cap=cv2.VideoCapture(0)

while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi = img[y:y + h, x:x + w]
        pred=model.predict_classes(np.expand_dims(resize(roi,(256,256))/255,0))

        if pred == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.rectangle(img, (x, y), (x + w, y - 70), (0, 255, 0), -1)
            cv2.putText(img, 'Mask Detected', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.rectangle(img, (x - 55, y), (x + w + 20, y - 70), (255, 0, 0), -1)
            cv2.putText(img, 'Mask Not Detected', (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


