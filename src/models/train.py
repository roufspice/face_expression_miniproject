import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import plot_model

#model Loading
from CNN import model_littleVGG
from CNN import big_XCEPTION
from CNN import XCEPTION


num_classes = 4
input_shape =(48,48,1)

#dataset loading
#faces_data_train_val_test_x,y :NPY file 불러오기
from numpy import load
'''
x_train = load('final_data/split/x_train.npy')
x_val=load('final_data/split/x_val.npy')
x_test=load('final_data/split/x_test.npy')
y_train=load('final_data/split/y_train.npy')
y_val=load('final_data/split/y_val.npy')
y_test=load('final_data/split/y_test.npy')
'''


#-----------------------------------------model_cimpile / model_callback-------------------------------------
#------------------------------------------model parameters/compilpation-------------------------------------
model = model_littleVGG
#model = big_XCEPTION(input_shape, num_classes)
#model = XCEPTION(input_shape, num_classes)


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#save_path
base_path = ''


#callbacks
log_file_path = base_path + '~~.log'
csv_logger = tf.keras.callbacks.CSVLogger(log_file_path, append=False)
early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=100)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=25,verbose=1)
trained_models_path = base_path + ''

model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}-{val_loss:0.4f}.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_names,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


#model_fit
history = model.fit(x_train, y_train, batch_size=64,
                    epochs=400,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(x_val, y_val)
                    )

model.evaluate(x_test,y_test,batch_size=2)