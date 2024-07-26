import tensorflow as tf
from tensorflow.keras.layers import *

kernel_initializer = 'he_uniform'
interpolation="nearest"

def conv_block_2D(x, filters, block_type, repeat=1,dilation_rate=1, size=3, padding='same'):
    result = x

    for i in range(0, repeat):
        
        if block_type == 'resnet':
            result = resnet_conv2D_block(result, filters, dilation_rate)
        elif block_type == 'conv':
            result = Conv2D(filters ,(3,3), activation='relu' , padding='same' ,kernel_initializer=kernel_initializer)(result)
        elif block_type == 'double_convolution':
            result = double_convolution_with_batch_normalization(result, filters)

        else:
            return None

    return result

def resnet_conv2D_block(x, filters, dilation_rate=1):
    x1 = Conv2D(filters, (1, 1), activation=None, kernel_initializer='he_uniform', padding='same',
                dilation_rate=dilation_rate, use_bias=False)(x)
    x1 = BatchNormalization(axis=-1)(x1)

    x = Conv2D(filters, (3, 3), activation=None, kernel_initializer='he_uniform', padding='same',
               dilation_rate=dilation_rate, use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), activation=None, kernel_initializer='he_uniform', padding='same',
               dilation_rate=dilation_rate, use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    
    x_final = add([x, x1])
    x_final = Activation('relu')(x_final)
    
    return x_final

def double_convolution_with_batch_normalization(x, filters):
    x = Conv2D(filters, (3, 3), activation=None, padding='same', kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), activation=None, padding='same', kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x
