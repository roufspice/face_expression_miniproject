# face_expression_miniproject

Face emotional expression classification using fer2013 datasets and images which is collected on Google images by crawling with a Tensorflow.Keras CNN models (XCEPTION and VGG16) and openCV

-  fer2013 + myDataset(google crawling images) facial expression test accuracy: 69.52%
-  fer2013 + myDataset(google crawling images) facial expression test accuracy(Hard voting): 77.13%



Referenced models are here: https://github.com/oarriaga/face_classification/blob/master/src/models/cnn.py
Using CNN model Structure
1) big_XCEPTION from referenced models(above github link)



+ DataSet
1) fer2013.csv : https://www.kaggle.com/nicolejyt/facialexpressionrecognition
2) googleCrawling dataSet(sample:1264 images) : 



# face_expression_examples:
---------------------------
<div>
<img src="https://github.com/roufspice/face_expression_miniproject/blob/master/images/angry_01.jpg" width="50%"></img>
<img src="https://github.com/roufspice/face_expression_miniproject/blob/master/images/neurtral_01.jpg" width="50%"></img>
</div>
<p>facial_expression_demo_Test</p>
<div>
<img src="https://github.com/roufspice/face_expression_miniproject/blob/master/images/openCV_demo.gif"></img>
</div>


### Run real-time emotion demo(실시간 시연 영상 코드):


### To train models for facial expression classification:




