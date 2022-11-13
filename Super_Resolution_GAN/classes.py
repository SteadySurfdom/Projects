import tensorflow as tf
# from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19

class GenBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(GenBlock,self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(kernel_size=3,filters=64,strides=1,padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(kernel_size=3,filters=64,strides=1,padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.prelu = tf.keras.layers.PReLU()
    
    def call(self,x,training = False):
        skip = x
        x = self.conv_1(x,training=training)
        x = self.bn_1(x,training=training)
        x = self.prelu(x,training=training)
        
        x = self.conv_2(x,training=training)
        x = self.bn_2(x,training=training) 
        return x + skip
    

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.gen_chain = tf.keras.Sequential([GenBlock() for i in range(16)])
        self.prelu_1 = tf.keras.layers.PReLU()
        self.prelu_2 = tf.keras.layers.PReLU()
        self.conv9641 = tf.keras.layers.Conv2D(kernel_size=9,filters=64,strides=1,padding='same',activation=tf.keras.layers.PReLU())
        self.conv3641 = tf.keras.layers.Conv2D(kernel_size=3,filters=64,strides=1,padding='same')
        self.conv32561_1 = tf.keras.layers.Conv2D(kernel_size=3,filters=256,strides=1,padding='same')
        self.conv32561_2 = tf.keras.layers.Conv2D(kernel_size=3,filters=256,strides=1,padding='same')
        self.conv931 = tf.keras.layers.Conv2D(kernel_size=9,filters=3,strides=1,padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
    
    def call(self,x,training = False):
        x = self.conv9641(x,training=training)
        
        skip = x
        x = self.gen_chain(x,training=training)
        
        x = self.conv3641(x,training=training)
        x = self.bn(x,training=training)
        x = x + skip

        x = self.conv32561_1(x,training=training)
        x = tf.nn.depth_to_space(input=x,block_size=2)
        x = self.prelu_1(x,training=training)
        
        x = self.conv32561_2(x,training=training)
        x = tf.nn.depth_to_space(input=x,block_size=2)
        x = self.prelu_2(x,training=training)

        x = self.conv931(x,training=training)
        return x
    
    def model(self):
        x = tf.keras.layers.Input(shape=(32,32,3))
        return tf.keras.Model(x,self.call(x))


class DisBlock(tf.keras.layers.Layer):
    def __init__(self,filters,stride,bn=True):
        super(DisBlock,self).__init__()
        self.block = tf.keras.Sequential()

        self.block.add(tf.keras.layers.Conv2D(kernel_size=3,filters=filters,strides=stride,padding='same'))
        # Adding the BatchNormalisation layer conditionally
        if bn:
            self.block.add(tf.keras.layers.BatchNormalization())
        self.block.add(tf.keras.layers.LeakyReLU(0.2))
    
    def call(self,x,training=False):
        return self.block(x,training=training)
    

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis_chain = tf.keras.Sequential()
        
        self.dis_chain.add(DisBlock(filters=64,stride=1,bn=False))
        self.dis_chain.add(DisBlock(filters=64,stride=2))

        channels = 128
        # Alternating strides (1,2,1,2,1,2)
        # Doubling filters after 2 iterations (128,128,256256,512,512)
        for i in range(6):
            self.dis_chain.add(DisBlock(filters=channels,stride=1 + i%2))
            if i%2==1:
                channels = channels * 2

        self.flatten = tf.keras.layers.Flatten()
        self.dense1024 = tf.keras.layers.Dense(units=1024,activation=tf.keras.layers.LeakyReLU(0.2))
        self.dense1 = tf.keras.layers.Dense(units=1,activation='sigmoid')

    def call(self,x,training=False):
        x = self.dis_chain(x,training=training)
        
        x = self.flatten(x)
        x = self.dense1024(x,training=training) 
        x = self.dense1(x,training=training)
        
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(128,128,3))
        return tf.keras.Model(x,self.call(x))

class VGGloss:
    def __init__(self,choice):
        vgg = VGG19(include_top=False,weights='imagenet',input_shape=(128,128,3))
        vgg.trainable = False
        # Gives the output of block2_conv2 layer
        if choice=='22':
            self.model = vgg.layers[:6]
        # Gives the output of block5_conv4 layer
        elif choice=='54':
            self.model = vgg.layers[:21]
        self.mse = tf.keras.losses.MeanSquaredError()

    def __call__(self,super_res,high_res):
        # Preprocess super res and high res images before passing it into the VGG19 model
        super_res = tf.keras.applications.vgg19.preprocess_input(super_res)
        high_res = tf.keras.applications.vgg19.preprocess_input(high_res)
        for layer in self.model:
            super_res = layer(super_res)
            high_res = layer(high_res)
        # return the MSE between the feature maps obtained of high res and super res
        return self.mse(high_res,super_res)
