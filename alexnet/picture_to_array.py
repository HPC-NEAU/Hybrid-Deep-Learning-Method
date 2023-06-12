# -*- coding: utf-8 -*-
"""
@author: 709
"""

import os,glob
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# path='../data/AI_CHALLENGER_dataset/'

path='../data/AI_CHALLENGER_dataset'
data=['Apple_CedaRrust','Apple_Healthy','Apple_Scab',
      'Cherry_Healthy','Cherry_PowderyMildew',
      'Citrus_GreeningJune','Citrus_Healthy',
      'Corn_CercosporaZeaemaydisTehonAndDaniels','Corn_CurvulariaLeafSpotFungus','Corn_Healthy','Corn_PucciniaPolysora',
      'Grape_BlackMeaslesFungus','Grape_BlackRotFungus','Grape_Healthy','Grape_LeafBlightFungus',
      'Peach_BacterialSpot','Peach_Healthy',
      'Pepper_Healthy','Pepper_Scab',
      'Potato_EarlyBlightFungus','Potato_Healthy','Potato_LateBlightFungus',
      'Strawberry_Healthy','Strawberry_Scorch',
      'Tomato_EarlyBlightFungus','Tomato_Healthy','Tomato_LateBlightWaterMold','Tomato_LeafMoldFungus',
      'Tomato_PowderyMildew','Tomato_SeptoriaLeafSpotFungus','Tomato_SpiderMiteDamage','Tomato_TargetSpotBacteria',
      'Tomato_YLCVVirus' 
      ]

y=[]
x=[]

print(data)
print(len(data))


indexid=0
for val in data:
    fext=val.split('_')[-1]
    if(fext=='Healthy'):
        for picture in os.listdir(path+'/'+val):
            y.append(indexid)
            #image=Image.open(path+'/'+val+'/'+picture)
            img=cv2.imread(path+'/'+val+'/'+picture, cv2.IMREAD_UNCHANGED)
            img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
            img=np.array(img)
            x.append(img)
            #print(img.shape)
            
    else:
        for filename in os.listdir(path+'/'+val):
            for picture in os.listdir(path+'/'+val+'/'+filename):
                y.append(indexid)
                #image=Image.open(path+'/'+val+'/'+filename+'/'+picture)
                img=cv2.imread(path+'/'+val+'/'+filename+'/'+picture, cv2.IMREAD_UNCHANGED)
                img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
                img=np.array(img)
                x.append(img)
                #print(img.shape)
   
    indexid=indexid+1
    print(indexid)
    x1=np.array(x)
    y1=np.array(y)
    print(x1.shape)
    print(y1.shape)

x=np.array(x)
y=np.array(y)
print(x.shape)
y=to_categorical(y,num_classes=len(data))
print(y.shape)

np.save(path+'/x.npy',x)
np.save(path+'/y.npy',y)


'''
for filename in os.listdir(path+'/'+data[0]):
    for picture in os.listdir(path+'/'+data[0]+'/'+filename):
        y.append(0)
        image=Image.open(path+'/'+data[0]+'/'+filename+'/'+picture)
        image=np.array(image)
        x.append(image)
        print(image.shape)
        
for picture in os.listdir(path+'/'+data[1]):
    y.append(1)
    image=Image.open(path+'/'+data[1]+'/'+picture)
    image=np.array(image)
    x.append(image)
    print(image.shape)


#读入图像 并统一图像尺寸
for filename in os.listdir(path+'/'+data[0]):
    for picture in os.listdir(path+'/'+data[0]+'/'+filename):
        y.append(0)
        #image=Image.open(path+'/'+data[0]+'/'+filename+'/'+picture)
        img=cv2.imread(path+'/'+data[0]+'/'+filename+'/'+picture, cv2.IMREAD_UNCHANGED)
        img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
        img=np.array(img)
        x.append(img)
        print(img.shape)
      
for picture in os.listdir(path+'/'+data[1]):
    y.append(1)
    img=cv2.imread(path+'/'+data[1]+'/'+picture, cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
    img=np.array(img)
    x.append(img)
    print(img.shape)


x=np.array(x)
y=np.array(y)
print(x.shape)
y=to_categorical(y,num_classes=len(data))
print(y.shape)


#np.save(path+'/x.npy',x)
#np.save(path+'/y.npy',y)

'''


#img = cv2.imread('./Pictures/python.png', cv2.IMREAD_UNCHANGED)

