# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:54:49 2020

@author: 709
"""

# from keras.applications import InceptionV3
# import numpy as np
# from keras.applications import ResNet50,InceptionResNetV2
# from keras.preprocessing.image import ImageDataGenerator
# from keras import layers
# from keras.optimizers import RMSprop,Adadelta
# from keras import activations
# from keras.models import load_model,Model
# from keras import callbacks
# from keras import regularizers
# import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
# import os
# import datetime
# from keras.models import load_model
# model_path='D:/zhou/model/h5/res50_top_best.h5'
# path='D:/AI CHALLENGER _dataset/corn/'
# x_validata=np.load(path+'x_validata.npy')
# y_validata=np.load(path+'y_validata.npy')
# model=load_model(model_path)
# # loss,acc=model.evaluate(x_validata,y_validata)
# print(model.evaluate(x_validata,y_validata))


import numpy as np
from keras.applications import ResNet50,InceptionResNetV2
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

path='/zhouchangjian_01/zcj/data/AI_CHALLENGER_dataset/'
path2='/zhouchangjian_01/zcj/classification/inception/'
x_train=np.load(path+'x_train.npy')
y_train=np.load(path+'y_train.npy')
x_test=np.load(path+'x_test.npy')
y_test=np.load(path+'y_test.npy')
# log_dir="G:/tensorBoard_log"
# log_dir = os.path.join(
#     "logs",
#     "fit",
#     datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
# )
# callbacks=[callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,embeddings_freq=1)]
# train_path='G:/Ai图片/AI CHALLENGER _dataset/Apple/train'
# validation_path='G:/Ai图片/AI CHALLENGER _dataset/Apple/validation'
# train_generator=ImageDataGenerator(rescale=1./255,
#                                     rotation_range=40,
#                                     width_shift_range=0.2,
#                                     height_shift_range=0.2,
#                                     shear_range=0.2,
#                                     zoom_range=0.2
#                                    )
# validation_generator=ImageDataGenerator(rescale=1./255)
# train_generator=train_generator.flow_from_directory(train_path,
#                                                     batch_size=16,
#                                                     target_size=(196,196),
#                                                     classes=['Apple_CedaRrust','Apple_Healthy','Apple_Scab'])
# validation_generator=validation_generator.flow_from_directory(validation_path,
#                                                               batch_size=16,
#                                                               target_size=(196,196),
#                                                               classes=['Apple_CedaRrust','Apple_Healthy','Apple_Scab'])
input_=layers.Input((196,196,3))

v2=InceptionResNetV2(weights=None,
               include_top=False,
               input_shape=(196,196,3))
'''
v2=InceptionResNetV2(weights='imagenet',
               include_top=False,
               input_shape=(196,196,3))
'''
v2.trainable=True#221 321 331
# for layer in res50.layers:
#     if layer.name=='conv1_conv':
#         layer.trainable=True
#         print(layer.name+" is trainable")
#     # if layer.name=='conv2_block1_0_conv':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv2_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
#     # if layer.name=='conv2_block1_1_relu':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv3_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
#     # if layer.name=='conv2_block1_2_relu':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv3_block3_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
x=v2(input_)
v2.summary()
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x) #l2范数正则化，系数0.002，没有过拟合，最后5次准确率91%+-1.5%
x=layers.Dropout(0.4)(x)
output=layers.Dense(33,activation='softmax')(x)
model=Model(input_,output)
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])# 等sgd 训练后，尝试Adadelta以及牛顿动量法
#res50.summary()
# plot_model(v2,to_file="D:/Experiment/model/model_struct/v2.png",show_shapes=True,show_layer_names=True,rankdir="TB")
history=model.fit(x_train,y_train,batch_size=16,epochs=200,validation_data=(x_test,y_test))
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
plt.title('Training and Validation accuracy of InceptionResNet-V2')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'nceptionResNet-v2_acc.png')
plt.clf()
plt.plot(epochs,loss,'r',label='train_loss')
plt.plot(epochs,val_loss,'b',label='validata_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Training and Validation loss of InceptionResNet-V2')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'InceptionResNet-v2_loss.png')
model.save(path2+'model_InceptionResNet-v2.h5')