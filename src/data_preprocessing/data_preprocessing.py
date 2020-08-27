import numpy as np
import pandas as pd
import cv2
import tensorflow as tf


#faces_data_array :fer 2013불러오기
faces = pd.read_csv('fer2013.csv')

#label_ disgust, fear, surprise dropping
disgust=faces[faces["emotion"]==1].index
faces=faces.drop(disgust)
fear=faces[faces["emotion"]==2].index
faces=faces.drop(fear)
surprise=faces[faces["emotion"]==5].index
faces=faces.drop(surprise)

#modifying labels: Happy: 3->0, Angry: 0->1, Sad: 4->2, Neutral: 6->3
faces["emotion"]=faces["emotion"].apply(lambda x:1 if x==0 else 0 if x==3 else 2 if x==4 else 3 if x==6 else 9 )

#save into csv
faces.to_csv('../fer2013_4.csv')

#load 'fer2013_4.csv'
df_new = pd.read_csv('fer2013_4.csv')

#csv to numpy.array(None,48,48,1)
pixels = df_new['pixels'].tolist()

faces=[]

for i in pixels:
  face = [int(pixel) for pixel in i.split(' ')]
  face = np.array(face)
  face = face.reshape(48,48)
  face = cv2.resize(face.astype('uint8'), (48,48))
  faces.append(face.astype('float32'))

#faces
faces=np.array(faces)
faces = np.expand_dims(faces,-1)
print(faces.shape)

#emotions
emotions = df_new["emotion"].values
emotions
emotions=np.array(emotions)

#save to numpy file
#np.save('emotions.npy',emotions)
#np.save('faces.npy',faces)

