import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation,Dropout,Lambda
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.initializers import TruncatedNormal,Constant
from keras.regularizers import l2
from matplotlib import pyplot as pyplot
from sklearn.metrics import confusion_matrix


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape
def getmodel():
    model=Sequential()
    model.add(Conv2D(24,(5,5),strides=(2,2),kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_shape=(66,200,3),use_bias=True))
    model.add(Conv2D(36,(5,5),strides=(2,2),kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_shape=(31,98,24),use_bias=True))
    model.add(Conv2D(48,(5,5),strides=(2,2),kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_shape=(14,47,36),use_bias=True))
    model.add(Conv2D(64,(3,3),strides=(1,1),kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_shape=(5,22,48),use_bias=True))
    model.add(Conv2D(64,(3,3),strides=(1,1),kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_shape=(1,18,64),use_bias=True))
    model.add(Flatten())
    model.add(Dense(units=1164,kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_dim=1152))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=100,kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_dim=1164))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=50,kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_dim=100))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10,kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),activation='relu',input_dim=50))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),input_dim=10))
    # model.add(Dense(units=4,kernel_initializer=TruncatedNormal(stddev=0.1),bias_initializer=Constant(value=0.1),input_dim=10,activation='softmax'))
    #model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "atan_0"))
    return model
