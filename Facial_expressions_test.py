import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import collections

import time

'''얼굴인식모델'''
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
'''감정분류모델'''
emotion_vgg16 = load_model('files/_littleVGG16_.14-0.68-1.5656.h5', compile=False)
emotion_bigX = load_model('files/_big_XCEPTION2_23-0.60-1.6831.hdf5',compile=False)
emotion_xcep = load_model('files/XCEPTION.07-0.61-1.0051.h5',compile=False)

'''감정도출'''
EMOTIONS = ["happy", "angry","sad","neutral"]

camera = cv2.VideoCapture(0)
while camera.isOpened():

    success, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5
                                            #minSize=(20,20)
                                            )
    # if faces are found, it returns the positions of detected faces as Rect(x,y,w,h)
    canvas = np.zeros((480, 640, 3), dtype="uint8")
    print(len(faces))
    if len(faces) > 0:
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        print(face)
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network
        print(gray.shape) #gray.shape= (480,640)
        roi = gray[fY:fY + fH, fX:fX + fW]
        print(roi.shape)  #300,300
        #학습된 모델 input_shape 맞추기

        roi = cv2.resize(roi, (48, 48))
        print(roi.shape)  #100,100

        #roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  #(1,100,100,1)
        print(roi.shape)







        # Emotion predict
        preds_vgg16 = emotion_vgg16.predict(roi)[0]
        preds_bigX = emotion_bigX.predict(roi)[0]
        preds_xcep = emotion_xcep.predict(roi)[0]

        #roi == x_test와 같음.
        print(preds_vgg16)
        print(preds_bigX)
        print(preds_xcep)
        emotion_probability_vgg16 = np.max(preds_vgg16)
        emotion_probability_bigX = np.max(preds_bigX)
        emotion_probability_xcep = np.max(preds_xcep)
        print(emotion_probability_vgg16)
        vgg16_result=preds_vgg16.argmax()
        bigX_result=preds_bigX.argmax()
        xcep_result=preds_xcep.argmax()

        print(vgg16_result)
        print(bigX_result)
        print(xcep_result)

        '''voting_func'''
        vot_list=[vgg16_result,bigX_result,xcep_result]

        if collections.Counter(vot_list).most_common()[0][1] >=2:
            preds_ = collections.Counter(vot_list).most_common()[0][0]
        else:
            preds_ = vgg16_result

        label = EMOTIONS[preds_vgg16.argmax()]
        print(label)

        #assign labeling
        if preds_ ==0:

            cv2.putText(frame,
                        label,
                        (fX , fY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 0),
                        2)
            cv2.rectangle(frame,
                          (fX, fY), #사각형의 시작점
                          (fX + fW, fY + fH), #시작점과 대각선에 있는 사각형의 끝점
                          (255, 255, 0), #사각형 색
                          3 #선굵기(default =1), -1이면 사각형 내부가 채워짐
                          )
        if preds_==1:

            cv2.putText(frame,
                        label,
                        (fX , fY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        2)
            cv2.rectangle(frame,
                          (fX, fY), #사각형의 시작점
                          (fX + fW, fY + fH), #시작점과 대각선에 있는 사각형의 끝점
                          (0, 0, 255), #사각형 색
                          3 #선굵기(default =1), -1이면 사각형 내부가 채워짐
                          )
        elif preds_ ==2:
            cv2.putText(frame,
                        label,
                        (fX, fY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        2)
            cv2.rectangle(frame,
                          (fX, fY),  # 사각형의 시작점
                          (fX + fW, fY + fH),  # 시작점과 대각선에 있는 사각형의 끝점
                          (255, 0, 0),  # 사각형 색
                          3  # 선굵기(default =1), -1이면 사각형 내부가 채워짐
                          )
        elif preds_ == 3:
            cv2.putText(frame,
                        label,
                        (fX, fY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        2)
            cv2.rectangle(frame,
                          (fX, fY),  # 사각형의 시작점
                          (fX + fW, fY + fH),  # 시작점과 대각선에 있는 사각형의 끝점
                          (255, 255, 255),  # 사각형 색
                          3  # 선굵기(default =1), -1이면 사각형 내부가 채워짐
                          )


        #label printing




    if success:
        #프레임출력
        cv2.imshow('Camera Window', frame)

        #'ESC'를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if (key==27):
            break

camera.release()
cv2.destroyALLWindows()
