#!/usr/bin/env python3

import sys
import numpy as np
import pathlib
import UTTTGame
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.layers import Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model, load_model

class neuralnetwork:
    CONV_LAYER_NUM_FILTER = 64 #agz is 256
    NUM_RES_LAYERS = 19 #agz is 19 or 39
    ADAM_LR=1e-3
    
    def __init__(self, path, init=True):
        self.path = path
        if init:
            self.nn = self.init_nn()
            self.compile_net()
        else:
            self.nn = self.load_net()
            self.compile_net()

    def load_net(self):
        return keras.models.load_model(str(self.path))

    def save_net(self, note=""):
        self.nn.save(str(self.path) + note)

    def compile_net(self):
        self.nn.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                        optimizer=Adam(self.ADAM_LR))
        
    def init_nn(self):
        input_shape = (9,9,UTTTGame.utttgame.NUMBER_OF_SAVED_GAME_STATES*2+1)
        inputs = Input(shape=input_shape)
        x = self.conv_layer(inputs)
        for i in range(self.NUM_RES_LAYERS):
            x = self.res_layer(x)
        pol = self.policy_head(x)
        val = self.value_head(x)
        model = Model(inputs = inputs, outputs = [pol, val])
        return model
    
    def conv_layer(self,
                   inputs,
                   num_filters=CONV_LAYER_NUM_FILTER,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4),
                   activation='relu'):
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))
        x = inputs
        x = conv(x)
        x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        return x

    def res_layer(self,
                  inputs):
        x = inputs
        x = self.conv_layer(x)
        x = self.conv_layer(x, activation=None)
        x = keras.layers.add([x, inputs])
        x = Activation('relu')(x)
        return x

    def policy_head(self,
                    inputs):
        x = inputs
        x = self.conv_layer(x,
                            num_filters = 2,
                            kernel_size = 1)
        x = Flatten()(x)
        x = Dense(81)(x)
        return x

    def value_head(self,
                   inputs):
        x = inputs
        x = self.conv_layer(x,
                            num_filters = 1,
                            kernel_size = 1)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation='tanh')(x)
        return x
        
    def evaluate(self,
             eval_input):
        policy, value = self.nn.predict(eval_input)
        return policy, value
        
def main():
    game = UTTTGame.utttgame()
    nn = neuralnetwork("test")
    #print(nn.nn.summary())
    pol, val = nn.evaluate(game.get_convnet_input().reshape(-1,9,9,9))
    print(pol, val)
    
    #from keras.utils import plot_model
    #import os
    #os.environ["PATH"] += os.pathsep + 'C:/Users/Tobi/Downloads/graphviz-2.38/release/bin'
    #plot_model(nn.nn, to_file='model.png')
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
