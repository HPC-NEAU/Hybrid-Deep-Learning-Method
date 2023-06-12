# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:19:21 2020

@author: VULCAN
"""
import numpy as np
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.optimizers import RMSprop
from keras import activations
from keras.models import load_model,Model
from keras import callbacks
from keras import regularizers
import matplotlib.pyplot as plt
import os
import datetime

log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
x_train=np.load("D:/AI CHALLENGER _dataset/apple/x_train.npy")
y_train=np.load("D:/AI CHALLENGER _dataset/apple/y_train.npy")
x_test=np.load("D:/AI CHALLENGER _dataset/apple/x_test.npy")
y_test=np.load("D:/AI CHALLENGER _dataset/apple/y_test.npy")

callbacks=[callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,embeddings_freq=1)]
input_=layers.Input((196,196,3))
res50=ResNet50(weights='imagenet',
               include_top=False,)
res50.trainable=False
for layer in res50.layers:
	if layer.name=='conv5_block3_2_conv':
		layer.trainable=True
	if layer.name=='conv5_block3_3_conv':
		layer.trainable=True
x=res50(input_)
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.0015))(x)
output=layers.Dense(3,activation='softmax')(x)
model=Model(input_,output)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=150,batch_size=32,validation_data=(x_test,y_test))
history=history.history
acc=history['acc']
val_acc=history['val_acc']
np.save("res50_15_acc.npy",acc)
np.save("res50_15_val_acc.npy",val_acc)
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='acc')
plt.plot(epochs,val_acc,'b',label='val_acc')
plt.show()
plt.savefig("res50_15.png")
model.save("Res50_15.h5")
