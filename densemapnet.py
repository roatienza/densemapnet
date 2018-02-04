'''DenseMapNet - a tiny network for fast disparity estimation
from stereo images

DenseMapNet class is where the actual model is built

Atienza, R. "Fast Disparity Estimation using Dense Networks".
International Conference on Robotics and Automation,
Brisbane, Australia, 2018.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Dense, Dropout
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D 
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D

import numpy as np
from utils import Settings


class DenseMapNet(object):
    def __init__(self, settings):
        self.settings =settings
        self.xdim = self.settings.xdim 
        self.ydim = self.settings.ydim
        self.channels = self.settings.channels
        self.model = None

    def build_model(self):
        if self.model:
            return self.model

        dropout = 0.2

        shape=(None, self.ydim, self.xdim, self.channels)
        left = Input(batch_shape=shape)
        right = Input(batch_shape=shape)

        # left image as reference
        x = Conv2D(filters=16, kernel_size=5, padding='same')(left)
        xleft = Conv2D(filters=1,
                       kernel_size=5,
                       padding='same',
                       dilation_rate=2)(left)

        # left and right images for disparity estimation
        xin = keras.layers.concatenate([left, right])
        xin = Conv2D(filters=32, kernel_size=5, padding='same')(xin)

        # image reduced by 8
        x8 = MaxPooling2D(8)(xin)
        x8 = BatchNormalization()(x8)
        x8 = Activation('relu', name='downsampled_stereo')(x8)

        dilation_rate = 1
        y = x8
        # correspondence network
        # parallel cnn at increasing dilation rate
        for i in range(4):
            a = Conv2D(filters=32,
                       kernel_size=5,
                       padding='same',
                       dilation_rate=dilation_rate)(x8)
            a = Dropout(dropout)(a)
            y = keras.layers.concatenate([a, y])
            dilation_rate += 1

        dilation_rate = 1
        x = MaxPooling2D(8)(x)
        # disparity network
        # dense interconnection inspired by DenseNet
        for i in range(4):
            x = keras.layers.concatenate([x, y])
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv2D(filters=64,
                       kernel_size=1,
                       padding='same')(y)

            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(filters=16,
                       kernel_size=5,
                       padding='same',
                       dilation_rate=dilation_rate)(y)
            y = Dropout(dropout)(y)
            dilation_rate += 1
        
        # disparity estimate scaled back to original image size
        x = keras.layers.concatenate([x, y], name='upsampled_disparity')
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        x = UpSampling2D(8)(x)
        x = ZeroPadding2D(padding=(2, 0))(x)

        # left image skip connection to disparity estimate
        x = keras.layers.concatenate([x, xleft])
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(filters=16, kernel_size=5, padding='same')(y)

        x = keras.layers.concatenate([x, y])
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2DTranspose(filters=1, kernel_size=9, padding='same')(y)

        # prediction
        yout = Activation('sigmoid', name='disparity_output')(y)

        # densemapnet model
        self.model = Model([left, right],yout)
       
        if self.settings.model_weights:
            print("Loading checkpoint model weights %s...."
                  % self.settings.model_weights)
            self.model.load_weights(self.settings.model_weights)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=RMSprop(lr=1e-3,decay=1e-6))
        print("DenseMapNet Model:")
        self.model.summary()

        return self.model
