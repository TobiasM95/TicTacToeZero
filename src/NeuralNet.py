import sys
import numpy as np
import pathlib
import UTTTGame
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model

class neuralnetwork:
    CONV_LAYER_NUM_FILTER = 16 #agz is 256
    NUM_RES_LAYERS = 5 #agz is 19 or 39
    ADAM_LR=1e-3
    
    def __init__(self):
        self.nn = self.init_nn()
        self.nn.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                        optimizer=Adam(self.ADAM_LR))

    def init_nn(self):
        input_shape = np.zeros((9,9,UTTTGame.utttgame.NUMBER_OF_SAVED_GAME_STATES*2+1)).shape
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
        x = Dense(81)(x)
        return x

    def value_head(self,
                   inputs):
        x = inputs
        x = self.conv_layer(x,
                            num_filters = 1,
                            kernel_size = 1)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation='tanh')(x)
        return x
        

def main():
    nn = neuralnetwork()
    print(nn.nn.summary())
    
    #from keras.utils import plot_model
    #import os
    #os.environ["PATH"] += os.pathsep + 'C:/Users/Tobi/Downloads/graphviz-2.38/release/bin'
    #plot_model(nn.nn, to_file='model.png')
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
