import keras
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras import layers
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, Lambda, \
    Conv2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import plot_model
import numpy as np
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
import time

layers=keras.layers,
models=keras.models,
utils=keras.utils

import keras
import tensorflow as tf

#config = tf.ConfigProto()
#path='D:/zhou/Data/jpp/'
path='/zhouchangjian_01/zcj/data/AI_CHALLENGER_dataset/'
path2='/zhouchangjian_01/zcj/classification/alexnet/'
x_train=np.load(path+'x_train.npy')
y_train=np.load(path+'y_train.npy')
x_validata=np.load(path+'x_validata.npy')
y_validata=np.load(path+'y_validata.npy')
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      # TensorFlow按需分配显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定显存分配比例
#sess=tf.compat.v1.Session(config=config)

def Alexnet():
    #input_tensor = Input(shape=(128, 128, 3))
    model = Sequential()

    # 第一层卷积层：42 个卷积核，大小为 5∗5, relu 激活函数
    model.add(Conv2D(42, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid', input_shape=(196, 196, 3)))
    # 第二层池化层：最大池化，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第三层卷积层：74 个卷积核，大小为 5*5，relu 激活函数
    model.add(Conv2D(74, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    # 第四层池化层：最大池化，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第五层卷积层：74 个卷积核，大小为 5*5，relu 激活函数
    model.add(Conv2D(74, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
    # 第六层池化层：最大池化，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 将参数进行扁平化，在 LeNet5 中称之为卷积层，实际上这一层是一维向量，和全连接层一样
    model.add(Flatten())
    # 随机丢弃
    model.add(Dropout(0.2))
    # 输出层 输出5类，用 softmax 激活函数计算分类概率
    model.add(Dense(33, activation='softmax'))
    # 设置损失函数和优化器配置
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['acc'])
    print(model.summary())
    return model

# 数据增强
# train_data_gen = ImageDataGenerator(
#                                     rescale=1 / 255.,
#                                     samplewise_center=True,
#                                     samplewise_std_normalization=True,
#                                     width_shift_range=0.05,
#                                     height_shift_range=0.05,
#                                     fill_mode='reflect',
#                                     horizontal_flip=True,
#                                     vertical_flip=True
#                                     )
# test_data_gen = ImageDataGenerator(
#                                     rescale=1 / 255.,
#                                     samplewise_center=True,
#                                     samplewise_std_normalization=True,
#                                     width_shift_range=0.05,
#                                     height_shift_range=0.05,
#                                     fill_mode='reflect',
#                                     horizontal_flip=True,
#                                     vertical_flip=True
#                                     )
#输入训练及验证数据
# train_gen = train_data_gen.flow_from_directory(directory='D:/research/DataSet/plantvillage/train',
#                                                target_size=(128, 128), color_mode='rgb',
#                                                class_mode='categorical',
#                                                batch_size=16, shuffle=True, seed=222
#                                                )
# val_gen = test_data_gen.flow_from_directory(directory='D:/research/DataSet/plantvillage/test/test',
#                                             target_size=(128, 128), color_mode='rgb',
#                                             class_mode='categorical',
#                                             batch_size=16, shuffle=True, seed=222
#                                             )

#训练函数
ftvggmodel = Alexnet()
history=ftvggmodel.fit(x_train,y_train,batch_size=16,epochs=200,validation_data=(x_validata,y_validata))
history=history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']
#np.save('E:/apple/resnet/acc.npy',acc)
#np.save('E:/apple/resnet/val_acc.npy',val_acc)
#np.save('E:/apple/resnet/loss.npy',loss)
#np.save('E:/apple/resnet/val_loss.npy',val_loss)
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='train_accuarcy')
plt.plot(epochs,val_acc,'b',label='validata_accuarcy')
max_val_acc_index=np.argmax(val_acc)
plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
plt.annotate(show_max, xytext=(-40,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and Validation accuracy of AlexNet')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'Validation loss of AlexNet acc.png')
plt.clf()
plt.plot(epochs,loss,'r',label='train_loss')
plt.plot(epochs,val_loss,'b',label='validata_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Training and Validation loss of AlexNet')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'Training and Validation loss ofAlexNet loss.png')
ftvggmodel.save(path2+'ftvggmodel_AlexNet.h5')
model.save(path2+'model_AlexNet.h5')