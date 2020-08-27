import pandas as pd
import numpy as np
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
from numpy import load

#faces_decection_data_array_detec.npy :NPY file 불러오기
faces_fer2013 = load('.../faces_fer2013_48X48_26217.npy')
faces_google = load('.../faces_google_48X48_1264.npy')

#차원확장 3차원 ->4차원
faces_google=np.expand_dims(faces_google,axis=-1)
faces_google.shape

#정답지 불러오기
#faces_decection_data_array_detec.npy :NPY file 불러오기
emotions_fer2013 = load('.../emotion_fer2013_48X48_26217.npy')
emotions_google = load('.../emotions_google_48X48_1264.npy')

#----------------------------------------split-----------------------------------------------

#fer train,test,val나누기 (train:60% val:20% test:20%)
x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(faces_fer2013, emotions_fer2013, test_size=0.2, random_state=1234)
x_train_f, x_val_f, y_train_f, y_val_f = train_test_split(x_train_f, y_train_f, test_size=0.25, random_state=1234)

#google train,test,val나누기
x_train_g, x_test_g, y_train_g, y_test_g = train_test_split(faces_google, emotions_google, test_size=0.2, random_state=1234)
x_train_g, x_val_g, y_train_g, y_val_g = train_test_split(x_train_g, y_train_g, test_size=0.25, random_state=1234)

print(x_train_g.shape,x_val_g.shape,x_test_g.shape)



#------------------------------------------------------------------------------------------------
# Image Augmentation google_Dataset +3000
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Image Augmentation
image_generator = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.5,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'

)

augmet_size = 3000
face = []
emotion = []
faces = [x_train_g, x_val_g, x_test_g]
emotions = [y_train_g, y_val_g, y_test_g]

idx = 0

for i, j in zip(faces, emotions):
    print(i.shape)
    randidx = np.random.randint(i.shape[0], size=augmet_size)

    face_augmented = i[randidx].copy()
    emotions_aumgented = j[randidx].copy()

    face_augmented = image_generator.flow(face_augmented, np.zeros(augmet_size),
                                          batch_size=augmet_size, shuffle=False).next()[0]  # shuffle == false

    # 원래 데이터인 x_train 에 Image Augmentation 된 x_augmented 를 추가합니다.
    # k=0
    i = np.concatenate((i, face_augmented))
    print(i.shape)
    print("-------------------------")

    # face[k]=i
    j = np.concatenate((j, emotions_aumgented))
    face.append(i)
    emotion.append(j)
    # emotion[k]=j
    # k=k+1

    # print(i.shape)
    # print(j.shape)

print(face[0].shape, face[1].shape, face[2].shape)
print(emotion[0].shape, emotion[1].shape, emotion[2].shape)


x_train_g=face[0]
x_val_g=face[1]
x_test_g=face[2]
y_train_g=emotion[0]
y_val_g=emotion[1]
y_test_g=emotion[2]
#--------------------------------------------------------------------------------------------------
# 문제지 합치기 (26217+1264 = 27481장)
# google_crawling 파일을 어떻게 불려서 합칠까?

faces_f = [x_train_f, x_val_f, x_test_f]
faces_g = [x_train_g, x_val_g, x_test_g]
# faces_total=[x_train_total,x_val_total,x_test_total]
faces_total = []
for i, j in zip(faces_f, faces_g):
    k = np.concatenate((i, j))
    print(k.shape)
    faces_total.append(k)

# 정답지 합치기
emotions_g = [y_train_g, y_val_g, y_test_g]
emotions_f = [y_train_f, y_val_f, y_test_f]
# emotions_total=[y_train_total,y_val_total,y_test_total]
emotions_total = []
idx = 0
for i, j in zip(emotions_f, emotions_g):
    k = np.concatenate((i, j))
    print(k.shape)

    emotions_total.append(k)
    # idx+=1
print(emotions_total[0].shape)
print(faces_total[0].shape)


#----------------최종 합치기 --------------------------------#
y_train_total= emotions_total[0]
y_val_total = emotions_total[1]
y_test_total= emotions_total[2]
x_train_total= faces_total[0]
x_val_total = faces_total[1]
x_test_total= faces_total[2]



#faces_data_array :NPY file로 저장

np.save('.../split/y_train',y_train_total)
np.save('.../split/y_val',y_val_total)
np.save('.../split/y_test',y_test_total)
np.save('.../split/x_train',x_train_total)
np.save('.../split/x_val',x_val_total)
np.save('.../split/x_test',x_test_total)