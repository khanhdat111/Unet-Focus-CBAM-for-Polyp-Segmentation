import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential

from CustomLayers.ConvBlock2D import conv_block_2D
from AttentionLayers.CBAM import cbam_block


kernel_initializer = 'he_uniform'
interpolation = "nearest"

def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_chanels))

    print('Starting U-Net-CBAM')

    p1 = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(input_layer)
    p1cb = cbam_block(p1)
    p2 = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(p1cb)
    p2cb = cbam_block(p2)
    p3 = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(p2cb)
    p3cb = cbam_block(p3)
    p4 = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(p3cb)
    p4cb = cbam_block(p4)
    p5 = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(p4cb)
    p5cb = cbam_block(p5)

    t0 = conv_block_2D(input_layer, starting_filters, 'double_convolution', repeat=1)
    t0cb = cbam_block(t0)

    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0cb)
    s1 = add([l1i, p1cb])
    t1 = conv_block_2D(s1, starting_filters * 2, 'double_convolution', repeat=1)
    t1cb = cbam_block(t1)

    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1cb)
    s2 = add([l2i, p2cb])
    t2 = conv_block_2D(s2, starting_filters * 4, 'double_convolution', repeat=1)
    t2cb = cbam_block(t2)

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2cb)
    s3 = add([l3i, p3cb])
    t3 = conv_block_2D(s3, starting_filters * 8, 'double_convolution', repeat=1)
    t3cb = cbam_block(t3)

    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3cb)
    s4 = add([l4i, p4cb])
    t4 = conv_block_2D(s4, starting_filters * 16, 'double_convolution', repeat=1)
    t4cb = cbam_block(t4)

    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4cb)
    s5 = add([l5i, p5cb])
    
    t51 = conv_block_2D(s5, starting_filters * 32, 'resnet', repeat=2)
    t53 = conv_block_2D(t51, starting_filters * 16, 'resnet', repeat=2)
    
    #----------------------------------------------------------------------------------#

    l5o = UpSampling2D((2, 2), interpolation=interpolation)(t53)
    c4 = add([l5o, t4cb])
    c4 = cbam_block(c4)
    q4 = conv_block_2D(c4, starting_filters * 8, 'double_convolution', repeat=1)
    
    
    l4o = UpSampling2D((2, 2), interpolation=interpolation)(q4)
    c3 = add([l4o, t3cb])
    c3 = cbam_block(c3)
    q3 = conv_block_2D(c3, starting_filters * 4, 'double_convolution', repeat=1)
    
    l3o = UpSampling2D((2, 2), interpolation=interpolation)(q3)
    c2 = add([l3o, t2cb])
    c2 = cbam_block(c2)
    q6 = conv_block_2D(c2, starting_filters * 2, 'double_convolution', repeat=1)
    
    
    l2o = UpSampling2D((2, 2), interpolation=interpolation)(q6)
    c1 = add([l2o, t1cb])
    c1 = cbam_block(c1)
    q1 = conv_block_2D(c1, starting_filters, 'double_convolution', repeat=1)

    
    l1o = UpSampling2D((2, 2), interpolation=interpolation)(q1)
    c0 = add([l1o, t0cb])
    c0 = cbam_block(c0)
    z1 = conv_block_2D(c0, starting_filters, 'double_convolution', repeat=1)
    
    output = Conv2D(out_classes, (1, 1), activation='sigmoid')(z1)

    model = Model(inputs=input_layer, outputs=output)

    return model
