import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time

'''얼굴인식모델'''
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
'''감정분류모델'''
emotion_classifier = load_model('files/_littleVGG16_.14-0.68-1.5656.h5', compile=False)
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
        preds = emotion_classifier.predict(roi)[0]
        #roi == x_test와 같음.
        print(preds)
        emotion_probability = np.max(preds)
        print(emotion_probability)
        label = EMOTIONS[preds.argmax()]
        print(label)

        #assign labeling
        if preds.argmax() ==0:

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
        if preds.argmax() ==1:

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
        elif preds.argmax() ==2:
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
        elif preds.argmax() == 3:
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
