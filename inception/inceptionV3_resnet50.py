# --coding:utf-8--
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import PIL
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3_matt import InceptionV3, preprocess_input
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import tensorflow as tf
import datetime
import os
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 引入Tensorboard


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


# train_num = get_nb_files('/home/pandafish/AnacondaProjects/Inceptionv3/dataset_my/train')  2500
# print(train_num)
# input('wait...')

# 数据准备
IM_WIDTH, IM_HEIGHT = 299, 299  # InceptionV3指定的图片尺寸
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量

train_dir = 'G:/python/coffee_leaf/实验性数据\yy2k5y8mxg-1/rebuilt.BRACOL_coffee_leaf_ images_datasets/coffee-datasets/coffee-datasets/leaf/三分类RNB/split1/ImBalanced/train'  # 训练集数据
val_dir = 'G:/python/coffee_leaf/实验性数据\yy2k5y8mxg-1/rebuilt.BRACOL_coffee_leaf_ images_datasets/coffee-datasets/coffee-datasets/leaf/三分类RNB/split1/ImBalanced/val'  # 验证集数据
# output_model_file = 'D:/python/keras-bcnn-image/LatestInsectDatasets3/LatestInsectDatasets/split1/test1slipt1Imbalancedclass40epoch30InceptionV3Image.model'
nb_classes = 3 # 分多少类
nb_epoch =  1 # 训练的循环数
batch_size = 8 # GPU同时训练的图片张数

nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
nb_epoch = int(nb_epoch)  # epoch数量
batch_size = int(batch_size)

# 　图片生成器
train_datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,e
#     horizontal_flip=True
    rescale=1 / 255.,
    samplewise_center=True,
    samplewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='reflect',
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT), color_mode='rgb',
    batch_size=batch_size, class_mode='categorical',shuffle=True, seed=222)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')


# 添加新层
def add_new_last_layer(base_model, nb_classes):
    """
    添加最后的层
    输入
    base_model和分类数量
    输出
    新的keras的model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(base_model.input, predictions)
    return model


# 冻上NB_IV3_LAYERS之前的层
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# 设置网络结构
model = InceptionResNetV2(input_shape=(299,299,3),weights='imagenet', include_top=False)
model = add_new_last_layer(model, nb_classes)
setup_to_finetune(model)

cp = ModelCheckpoint('G:/python/coffee_leaf/model/inceptionV3/inc_res测试/train_ split1_30epochs_3class_inceptionV3.h5', monitor='accuracy', verbose=1,
                     save_best_only=True, save_weights_only=False,
                     mode='auto', period=1)

es = EarlyStopping(monitor='val_accuracy',
                   patience=6, verbose=1, mode='auto')
lr_reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.1, epsilon=1e-5, patience=2, verbose=1, min_lr=0.00000001)
tb=keras.callbacks.TensorBoard(log_dir='G:/python/coffee_leaf/logs/inceptionV3/inc_res测试/my_log_dir'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
callbackslist = [cp, es, lr_reduce,tb,]
# tf.gfile.DeleteRecursively('D:/python_try/logs/inceptionV3/昆虫图像处理/my_log_dir/')#每次运行清除之前的
# 模式二训练
history_ft = model.fit_generator(
    train_generator,
    verbose=1,
    # samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto1',
    callbacks=callbackslist,
    shuffle=True,
    )

# 模型保存
# model.save(output_model_file)


# 画图output_model_file
# def plot_training(history):
#     accuracy = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(len(accuracy))
#     plt.plot(epochs, accuracy, 'r.')
#     plt.plot(epochs, val_acc, 'r')
#     plt.title('Training and validation accuracy')
#     plt.figure()
#     plt.plot(epochs, loss, 'r.')
#     plt.plot(epochs, val_loss, 'r-')
#     plt.title('Training and validation loss')
#     plt.show()


# 训练的acc_loss图
# plot_training(history_ft)
