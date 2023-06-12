# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:48:45 2020

@author: 709
"""

from keras.models import load_model
import numpy as np
path='/zhouchangjian_01/zcj/data/AI_CHALLENGER_dataset/'
model_path='/zhouchangjian_01/zcj/classification/alexnet/ftvggmodel_AlexNet.h5'
x=np.load(path+'/'+'x_test.npy')
y=np.load(path+'/'+'y_test.npy')
model=load_model(model_path)
model.evaluate(x,y)