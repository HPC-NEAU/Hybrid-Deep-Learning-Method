# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:54:49 2020

@author: 709
"""

from keras.applications import InceptionV3
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from keras.applications import ResNet50,InceptionResNetV2,DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.optimizers import RMSprop,Adadelta
from keras import activations
from keras.models import load_model,Model
from keras import callbacks
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import datetime
from keras.losses import categorical_crossentropy
from keras.models import load_model
# import sys
# sys.path.append(r"C:\Users\709\Desktop\DenseNet-Keras-master")
# import  densenet121
path='D:/AI CHALLENGER _dataset/Tomato/'
#validation_path='D:/AI CHALLENGER _dataset/corn/'
x_train=np.load(path+'x_train.npy')
y_train=np.load(path+'y_train.npy')
x_validata=np.load(path+'x_validata.npy')
y_validata=np.load(path+'y_validata.npy')
DenseNet=DenseNet121(weights='imagenet',include_top=False,input_shape=(196,196,3))
#DenseNet=DenseNet121(weights=None,include_top=False,input_shape=(196,196,3))
DenseNet.trainable=False
for layer in DenseNet.layers:
    if layer.name=='conv2_block2_1_conv':
        layer.trainable=True
        print(layer.name+"is trainable")
    if layer.name=='conv1/conv':
        layer.trainable=True
        print(layer.name+"is trainable")
    if layer.name=='conv3_block3_1_conv':
        layer.trainable=True
        print(layer.name+"is trainable")
    if layer.name=='conv2_block3_1_conv':
        layer.trainable=True
        print(layer.name+"is trainable")
plot_model(DenseNet,'D:/Experiment/model/model_struct/DenseNet121.png',show_layer_names=True)
input_=layers.Input((196,196,3))
x=DenseNet(input_)
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.00005))(x)
x=layers.Dropout(0.5)(x)
output=layers.Dense(9,activation='softmax')(x)
model=Model(input_, output)
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,batch_size=16 ,epochs=2,validation_data=(x_validata,y_validata))
history=history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='train_accuarcy')
plt.plot(epochs,val_acc,'b',label='validata_accuarcy')
max_val_acc_index=np.argmax(val_acc)
plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
plt.annotate(show_max, xytext=(-20,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and Validation accuracy of DenseNet121 on Tomato dataset')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig('D:/picture/Dense/acc_DenseNet121_tomato.png')
plt.clf()
plt.plot(epochs,loss,'r',label='train_loss')
plt.plot(epochs,val_loss,'b',label='validata_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Training and Validation loss of DenseNet121 on Tomato dataset')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig('D:/picture/Dense/loss_DenseNet121_tomato.png')
#model.save('D:/Experiment/model/h5/DenseNet121.h5')
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
     saver.save(sess, 'D:/Experiment/model/dense.ckpt', global_step=epochs)
    





