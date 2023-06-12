# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:39:30 2020

@author: VULCAN
"""
from keras.models import Sequential,load_model
from PIL import Image
from keras.applications.resnet50 import decode_predictions
import numpy as np
#model_path='/zhouchangjian_01/zcj/model/h5/transfer/DenseNet121.h5'
model_path='/zhouchangjian_01/zcj/data/AI_CHALLENGER_dataset/rdn_240/rdn_240.h5'

model=load_model(model_path)
img_path='/zhouchangjian_01/zcj/data/AI_CHALLENGER_dataset/Cherry_PowderyMildew/Cherry_PowderyMildew_serious/00000125.jpg'
'''
classes=['Potato_EarlyBlightFungu','Potato_Healthy','Potato_LateBlightFungus','Tomato_EarlyBlightFungus','Tomato_Healthy','Tomato_LateBlightWaterMold','Tomato_LeafMoldFungus','Tomato_PowderyMildew','Tomato_SeptoriaLeafSpotFungus','Tomato_SpiderMiteDamage','Tomato_TargetSpotBacteria','Tomato_YLCVVirus']
'''
classes=['Apple_CedaRrust','Apple_Healthy','Apple_Scab','Cherry_Healthy','Cherry_PowderyMildew','Citrus_GreeningJune','Citrus_Healthy','Corn_CercosporaZeaemaydisTehonAndDaniels','Corn_CurvulariaLeafSpotFungus','Corn_Healthy','Corn_PucciniaPolysora','Grape_BlackMeaslesFungus','Grape_BlackRotFungus','Grape_Healthy','Grape_LeafBlightFungus',
'Peach_BacterialSpot','Peach_Healthy','Pepper_Healthy','Pepper_Scab','Potato_EarlyBlightFungus','Potato_Healthy','Potato_LateBlightFungus','Strawberry_Healthy',
'Strawberry_Scorch','Tomato_EarlyBlightFungus','Tomato_Healthy','Tomato_LateBlightWaterMold','Tomato_LeafMoldFungus','Tomato_PowderyMildew','Tomato_SeptoriaLeafSpotFungus','Tomato_SpiderMiteDamage','Tomato_TargetSpotBacteria','Tomato_YLCVVirus' ]
img=Image.open(img_path)
img=img.resize((196,196),Image.ANTIALIAS)
img_=np.array(img)
#print(img.shape)
#img=img.resize((196,196),Image.ANTIALIAS)
img=np.expand_dims(img_,axis=0)
history=model.predict(img)
history=history[0]
index=np.argmax(history)
probability=100*history[index]/np.sum(history)
print('this is belong to '+classes[index]+'\nthe probability is  %.4f'%(probability)+'%')
import matplotlib.pyplot as plt
plt.imshow(img_)
plt.title(classes[index]+": %.4f"%(probability)+"%")
plt.savefig("test3.png")
