import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import cv2

# load csv
df = pd.read_csv("../Dataset.csv")

#정답지만들기
df["answer"]=df["Facial_Expression"].apply(lambda x:0 if x=="happy" else(1 if x=="angry" else(2 if x=="sad" else 3)))
answer_list=["happy","angry","sad","neutral"]

# 파일쌓기
lis_48 = []
num = 0
for path in df["Unnamed: 0"].values:
    num += 1
    img_temp_path = '../' + str(path)
    img = cv2.imread(img_temp_path)
    img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_temp=img_temp/255.0
    img = cv2.resize(img_temp, (48, 48))
    if num % 20 == 0:
        print(num, "번째 완료/총1264개중")
    lis_48.append(img)

faces_48 = np.stack(lis_48)

#faces_data_array :NPY file로 저장
np.save('faces_google_48X48_1264.npy',faces_48)
np.save('emotions_google_48X48_1264.npy',np.array(df['emotion']))