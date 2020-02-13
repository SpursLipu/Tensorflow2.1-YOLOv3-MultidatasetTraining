# Author:LiPu
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
import utils
import numpy as np


class Darknet(Model):
    def __init__(self, image_size=416):
        super(Darknet, self).__init__()
        self.c1 = Conv2D(input_shape=(image_size, image_size, 3),
                         filters=32, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.ac1 = LeakyReLU(alpha=0.1)
        self.pad1 = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', strides=2, use_bias=False)
        self.bn2 = BatchNormalization()
        self.ac2 = LeakyReLU(alpha=0.1)

        # RB1
        self.RB1_conv1 = Conv2D(32, (1, 1), padding='same',
                                use_bias=False)
        self.RB1_bn1 = BatchNormalization()
        self.RB1_ac1 = LeakyReLU(alpha=0.1)
        self.RB1_conv2 = Conv2D(32 * 2, (3, 3), padding='same', use_bias=False)
        self.RB1_bn2 = BatchNormalization()
        self.RB1_ac2 = LeakyReLU(alpha=0.1)

        self.pad2 = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='valid', strides=2, use_bias=False)
        self.bn3 = BatchNormalization()
        self.ac3 = LeakyReLU(alpha=0.1)
        # RB2_1
        self.RB2_1_conv1 = Conv2D(64, (1, 1), padding='same', use_bias=False)
        self.RB2_1_bn1 = BatchNormalization()
        self.RB2_1_ac1 = LeakyReLU(alpha=0.1)
        self.RB2_1_conv2 = Conv2D(64 * 2, (3, 3), padding='same', use_bias=False)
        self.RB2_1_bn2 = BatchNormalization()
        self.RB2_1_ac2 = LeakyReLU(alpha=0.1)

        # RB2_2
        self.RB2_2_conv1 = Conv2D(64, (1, 1), padding='same', use_bias=False)
        self.RB2_2_bn1 = BatchNormalization()
        self.RB2_2_ac1 = LeakyReLU(alpha=0.1)
        self.RB2_2_conv2 = Conv2D(64 * 2, (3, 3), padding='same', use_bias=False)
        self.RB2_2_bn2 = BatchNormalization()
        self.RB2_2_ac2 = LeakyReLU(alpha=0.1)

        self.pad3 = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.c4 = Conv2D(filters=256, kernel_size=(3, 3), padding='valid', strides=2, use_bias=False)
        self.bn4 = BatchNormalization()
        self.ac4 = LeakyReLU(alpha=0.1)

        # RB3_1
        self.RB3_1_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_1_bn1 = BatchNormalization()
        self.RB3_1_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_1_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_1_bn2 = BatchNormalization()
        self.RB3_1_ac2 = LeakyReLU(alpha=0.1)
        # RB3_2
        self.RB3_2_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_2_bn1 = BatchNormalization()
        self.RB3_2_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_2_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_2_bn2 = BatchNormalization()
        self.RB3_2_ac2 = LeakyReLU(alpha=0.1)
        # RB3_3
        self.RB3_3_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_3_bn1 = BatchNormalization()
        self.RB3_3_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_3_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_3_bn2 = BatchNormalization()
        self.RB3_3_ac2 = LeakyReLU(alpha=0.1)
        # RB3_4
        self.RB3_4_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_4_bn1 = BatchNormalization()
        self.RB3_4_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_4_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_4_bn2 = BatchNormalization()
        self.RB3_4_ac2 = LeakyReLU(alpha=0.1)
        # RB3_5
        self.RB3_5_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_5_bn1 = BatchNormalization()
        self.RB3_5_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_5_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_5_bn2 = BatchNormalization()
        self.RB3_5_ac2 = LeakyReLU(alpha=0.1)
        # RB3_6
        self.RB3_6_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_6_bn1 = BatchNormalization()
        self.RB3_6_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_6_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_6_bn2 = BatchNormalization()
        self.RB3_6_ac2 = LeakyReLU(alpha=0.1)
        # RB3_7
        self.RB3_7_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_7_bn1 = BatchNormalization()
        self.RB3_7_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_7_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_7_bn2 = BatchNormalization()
        self.RB3_7_ac2 = LeakyReLU(alpha=0.1)
        # RB3_8
        self.RB3_8_conv1 = Conv2D(128, (1, 1), padding='same', use_bias=False)
        self.RB3_8_bn1 = BatchNormalization()
        self.RB3_8_ac1 = LeakyReLU(alpha=0.1)
        self.RB3_8_conv2 = Conv2D(128 * 2, (3, 3), padding='same', use_bias=False)
        self.RB3_8_bn2 = BatchNormalization()
        self.RB3_8_ac2 = LeakyReLU(alpha=0.1)

        self.pad4 = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.c5 = Conv2D(filters=512, kernel_size=(3, 3), padding='valid', strides=2, use_bias=False)
        self.bn5 = BatchNormalization()
        self.ac5 = LeakyReLU(alpha=0.1)
        # RB4_1
        self.RB4_1_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_1_bn1 = BatchNormalization()
        self.RB4_1_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_1_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_1_bn2 = BatchNormalization()
        self.RB4_1_ac2 = LeakyReLU(alpha=0.1)
        # RB4_2
        self.RB4_2_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_2_bn1 = BatchNormalization()
        self.RB4_2_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_2_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_2_bn2 = BatchNormalization()
        self.RB4_2_ac2 = LeakyReLU(alpha=0.1)
        # RB4_3
        self.RB4_3_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_3_bn1 = BatchNormalization()
        self.RB4_3_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_3_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_3_bn2 = BatchNormalization()
        self.RB4_3_ac2 = LeakyReLU(alpha=0.1)
        # RB4_4
        self.RB4_4_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_4_bn1 = BatchNormalization()
        self.RB4_4_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_4_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_4_bn2 = BatchNormalization()
        self.RB4_4_ac2 = LeakyReLU(alpha=0.1)
        # RB4_5
        self.RB4_5_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_5_bn1 = BatchNormalization()
        self.RB4_5_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_5_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_5_bn2 = BatchNormalization()
        self.RB4_5_ac2 = LeakyReLU(alpha=0.1)
        # RB4_6
        self.RB4_6_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_6_bn1 = BatchNormalization()
        self.RB4_6_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_6_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_6_bn2 = BatchNormalization()
        self.RB4_6_ac2 = LeakyReLU(alpha=0.1)
        # RB4_7
        self.RB4_7_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_7_bn1 = BatchNormalization()
        self.RB4_7_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_7_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_7_bn2 = BatchNormalization()
        self.RB4_7_ac2 = LeakyReLU(alpha=0.1)
        # RB4_8
        self.RB4_8_conv1 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.RB4_8_bn1 = BatchNormalization()
        self.RB4_8_ac1 = LeakyReLU(alpha=0.1)
        self.RB4_8_conv2 = Conv2D(256 * 2, (3, 3), padding='same', use_bias=False)
        self.RB4_8_bn2 = BatchNormalization()
        self.RB4_8_ac2 = LeakyReLU(alpha=0.1)

        self.pad5 = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.c6 = Conv2D(filters=1024, kernel_size=(3, 3), padding='valid', strides=2, use_bias=False)
        self.bn6 = BatchNormalization()
        self.ac6 = LeakyReLU(alpha=0.1)
        # RB5_1
        self.RB5_1_conv1 = Conv2D(512, (1, 1), padding='same', use_bias=False)
        self.RB5_1_bn1 = BatchNormalization()
        self.RB5_1_ac1 = LeakyReLU(alpha=0.1)
        self.RB5_1_conv2 = Conv2D(512 * 2, (3, 3), padding='same', use_bias=False)
        self.RB5_1_bn2 = BatchNormalization()
        self.RB5_1_ac2 = LeakyReLU(alpha=0.1)
        # RB5_2
        self.RB5_2_conv1 = Conv2D(512, (1, 1), padding='same', use_bias=False)
        self.RB5_2_bn1 = BatchNormalization()
        self.RB5_2_ac1 = LeakyReLU(alpha=0.1)
        self.RB5_2_conv2 = Conv2D(512 * 2, (3, 3), padding='same', use_bias=False)
        self.RB5_2_bn2 = BatchNormalization()
        self.RB5_2_ac2 = LeakyReLU(alpha=0.1)
        # RB5_3
        self.RB5_3_conv1 = Conv2D(512, (1, 1), padding='same', use_bias=False)
        self.RB5_3_bn1 = BatchNormalization()
        self.RB5_3_ac1 = LeakyReLU(alpha=0.1)
        self.RB5_3_conv2 = Conv2D(512 * 2, (3, 3), padding='same', use_bias=False)
        self.RB5_3_bn2 = BatchNormalization()
        self.RB5_3_ac2 = LeakyReLU(alpha=0.1)
        # RB5_5
        self.RB5_4_conv1 = Conv2D(512, (1, 1), padding='same', use_bias=False)
        self.RB5_4_bn1 = BatchNormalization()
        self.RB5_4_ac1 = LeakyReLU(alpha=0.1)
        self.RB5_4_conv2 = Conv2D(512 * 2, (3, 3), padding='same', use_bias=False)
        self.RB5_4_bn2 = BatchNormalization()
        self.RB5_4_ac2 = LeakyReLU(alpha=0.1)

    def call(self, x):
        output = self.c1(x)
        output = self.bn1(output)
        output = self.ac1(output)
        output = self.pad1(output)
        output = self.c2(output)
        output = self.bn2(output)
        output = self.ac2(output)

        # RB1
        shortcut_1 = output
        output = self.RB1_conv1(output)
        output = self.RB1_bn1(output)
        output = self.RB1_ac1(output)
        output = self.RB1_conv2(output)
        output = self.RB1_bn2(output)
        output = self.RB1_ac2(output)
        output = tf.add(output, shortcut_1)

        output = self.pad2(output)
        output = self.c3(output)
        output = self.bn3(output)
        output = self.ac3(output)

        # RB2_1
        shortcut_2_1 = output
        output = self.RB2_1_conv1(output)
        output = self.RB2_1_bn1(output)
        output = self.RB2_1_ac1(output)
        output = self.RB2_1_conv2(output)
        output = self.RB2_1_bn2(output)
        output = self.RB2_1_ac2(output)
        output = tf.add(output, shortcut_2_1)
        # RB2_2
        shortcut_2_2 = output
        output = self.RB2_2_conv1(output)
        output = self.RB2_2_bn1(output)
        output = self.RB2_2_ac1(output)
        output = self.RB2_2_conv2(output)
        output = self.RB2_2_bn2(output)
        output = self.RB2_2_ac2(output)
        output = tf.add(output, shortcut_2_2)

        output = self.pad3(output)
        output = self.c4(output)
        output = self.bn4(output)
        output = self.ac4(output)
        # RB3_1
        shortcut_3_1 = output
        output = self.RB3_1_conv1(output)
        output = self.RB3_1_bn1(output)
        output = self.RB3_1_ac1(output)
        output = self.RB3_1_conv2(output)
        output = self.RB3_1_bn2(output)
        output = self.RB3_1_ac2(output)
        output = tf.add(output, shortcut_3_1)
        # RB3_2
        shortcut_3_2 = output
        output = self.RB3_2_conv1(output)
        output = self.RB3_2_bn1(output)
        output = self.RB3_2_ac1(output)
        output = self.RB3_2_conv2(output)
        output = self.RB3_2_bn2(output)
        output = self.RB3_2_ac2(output)
        output = tf.add(output, shortcut_3_2)
        # RB3_3
        shortcut_3_3 = output
        output = self.RB3_3_conv1(output)
        output = self.RB3_3_bn1(output)
        output = self.RB3_3_ac1(output)
        output = self.RB3_3_conv2(output)
        output = self.RB3_3_bn2(output)
        output = self.RB3_3_ac2(output)
        output = tf.add(output, shortcut_3_3)
        # RB3_4
        shortcut_3_4 = output
        output = self.RB3_4_conv1(output)
        output = self.RB3_4_bn1(output)
        output = self.RB3_4_ac1(output)
        output = self.RB3_4_conv2(output)
        output = self.RB3_4_bn2(output)
        output = self.RB3_4_ac2(output)
        output = tf.add(output, shortcut_3_4)
        # RB3_5
        shortcut_3_5 = output
        output = self.RB3_5_conv1(output)
        output = self.RB3_5_bn1(output)
        output = self.RB3_5_ac1(output)
        output = self.RB3_5_conv2(output)
        output = self.RB3_5_bn2(output)
        output = self.RB3_5_ac2(output)
        output = tf.add(output, shortcut_3_5)
        # RB3_6
        shortcut_3_6 = output
        output = self.RB3_6_conv1(output)
        output = self.RB3_6_bn1(output)
        output = self.RB3_6_ac1(output)
        output = self.RB3_6_conv2(output)
        output = self.RB3_6_bn2(output)
        output = self.RB3_6_ac2(output)
        output = tf.add(output, shortcut_3_6)
        # RB3_7
        shortcut_3_7 = output
        output = self.RB3_7_conv1(output)
        output = self.RB3_7_bn1(output)
        output = self.RB3_7_ac1(output)
        output = self.RB3_7_conv2(output)
        output = self.RB3_7_bn2(output)
        output = self.RB3_7_ac2(output)
        output = tf.add(output, shortcut_3_7)
        # RB3_8
        shortcut_3_8 = output
        output = self.RB3_8_conv1(output)
        output = self.RB3_8_bn1(output)
        output = self.RB3_8_ac1(output)
        output = self.RB3_8_conv2(output)
        output = self.RB3_8_bn2(output)
        output = self.RB3_8_ac2(output)
        output = tf.add(output, shortcut_3_8)

        route_1 = output
        output = self.pad4(output)
        output = self.c5(output)
        output = self.bn5(output)
        output = self.ac5(output)
        # RB4_1
        shortcut_4_1 = output
        output = self.RB4_1_conv1(output)
        output = self.RB4_1_bn1(output)
        output = self.RB4_1_ac1(output)
        output = self.RB4_1_conv2(output)
        output = self.RB4_1_bn2(output)
        output = self.RB4_1_ac2(output)
        output = tf.add(output, shortcut_4_1)
        # RB4_2
        shortcut_4_2 = output
        output = self.RB4_2_conv1(output)
        output = self.RB4_2_bn1(output)
        output = self.RB4_2_ac1(output)
        output = self.RB4_2_conv2(output)
        output = self.RB4_2_bn2(output)
        output = self.RB4_2_ac2(output)
        output = tf.add(output, shortcut_4_2)
        # RB4_3
        shortcut_4_3 = output
        output = self.RB4_3_conv1(output)
        output = self.RB4_3_bn1(output)
        output = self.RB4_3_ac1(output)
        output = self.RB4_3_conv2(output)
        output = self.RB4_3_bn2(output)
        output = self.RB4_3_ac2(output)
        output = tf.add(output, shortcut_4_3)
        # RB4_4
        shortcut_4_4 = output
        output = self.RB4_4_conv1(output)
        output = self.RB4_4_bn1(output)
        output = self.RB4_4_ac1(output)
        output = self.RB4_4_conv2(output)
        output = self.RB4_4_bn2(output)
        output = self.RB4_4_ac2(output)
        output = tf.add(output, shortcut_4_4)
        # RB4_5
        shortcut_4_5 = output
        output = self.RB4_5_conv1(output)
        output = self.RB4_5_bn1(output)
        output = self.RB4_5_ac1(output)
        output = self.RB4_5_conv2(output)
        output = self.RB4_5_bn2(output)
        output = self.RB4_5_ac2(output)
        output = tf.add(output, shortcut_4_5)
        # RB4_6
        shortcut_4_6 = output
        output = self.RB4_6_conv1(output)
        output = self.RB4_6_bn1(output)
        output = self.RB4_6_ac1(output)
        output = self.RB4_6_conv2(output)
        output = self.RB4_6_bn2(output)
        output = self.RB4_6_ac2(output)
        output = tf.add(output, shortcut_4_6)
        # RB4_7
        shortcut_4_7 = output
        output = self.RB4_7_conv1(output)
        output = self.RB4_7_bn1(output)
        output = self.RB4_7_ac1(output)
        output = self.RB4_7_conv2(output)
        output = self.RB4_7_bn2(output)
        output = self.RB4_7_ac2(output)
        output = tf.add(output, shortcut_4_7)
        # RB4_8
        shortcut_4_8 = output
        output = self.RB4_8_conv1(output)
        output = self.RB4_8_bn1(output)
        output = self.RB4_8_ac1(output)
        output = self.RB4_8_conv2(output)
        output = self.RB4_8_bn2(output)
        output = self.RB4_8_ac2(output)
        output = tf.add(output, shortcut_4_8)
        route_2 = output
        output = self.pad5(output)
        output = self.c6(output)
        output = self.bn6(output)
        output = self.ac6(output)
        # RB5_1
        shortcut_5_1 = output
        output = self.RB5_1_conv1(output)
        output = self.RB5_1_bn1(output)
        output = self.RB5_1_ac1(output)
        output = self.RB5_1_conv2(output)
        output = self.RB5_1_bn2(output)
        output = self.RB5_1_ac2(output)
        output = tf.add(output, shortcut_5_1)
        # RB5_2
        shortcut_5_2 = output
        output = self.RB5_2_conv1(output)
        output = self.RB5_2_bn1(output)
        output = self.RB5_2_ac1(output)
        output = self.RB5_2_conv2(output)
        output = self.RB5_2_bn2(output)
        output = self.RB5_2_ac2(output)
        output = tf.add(output, shortcut_5_2)
        # RB5_3
        shortcut_5_3 = output
        output = self.RB5_3_conv1(output)
        output = self.RB5_3_bn1(output)
        output = self.RB5_3_ac1(output)
        output = self.RB5_3_conv2(output)
        output = self.RB5_3_bn2(output)
        output = self.RB5_3_ac2(output)
        output = tf.add(output, shortcut_5_3)
        # RB5_4
        shortcut_5_4 = output
        output = self.RB5_4_conv1(output)
        output = self.RB5_4_bn1(output)
        output = self.RB5_4_ac1(output)
        output = self.RB5_4_conv2(output)
        output = self.RB5_4_bn2(output)
        output = self.RB5_4_ac2(output)
        output = tf.add(output, shortcut_5_4)
        return route_1, route_2, output


# class RBlock(Model):
#     def __init__(self, filters):
#         super(RBlock, self).__init__()
#         self.filers = filters
#         self.conv1 = Conv2D(filters, (1, 1), padding='same', use_bias=False)
#         self.bn1 = BatchNormalization()
#         self.ac1 = LeakyReLU(alpha=0.1)
#         self.conv2 = Conv2D(filters * 2, (3, 3), padding='same', use_bias=False)
#         self.bn2 = BatchNormalization()
#         self.ac2 = LeakyReLU(alpha=0.1)
#
#     def call(self, x):
#         output = self.conv1(x)
#         output = self.bn1(output)
#         output = self.ac1(output)
#         output = self.conv2(output)
#         output = self.bn2(output)
#         output = self.ac2(output)
#         output = tf.add(output, x)
#         return output


class yolov3(Model):
    def __init__(self, opt):
        super(yolov3, self).__init__()
        # Darknet
        self.opt = opt
        self.image_size = self.opt.img_size
        self.num_class = len(utils.read_class_names(self.opt.data))
        self.Darknet = Darknet(self.image_size)
        # Convolution Set1
        self.c1_1 = Conv2D(input_shape=(13, 13, 1024), filters=512, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn1_1 = BatchNormalization()
        self.ac1_1 = LeakyReLU(alpha=0.1)
        self.c1_2 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn1_2 = BatchNormalization()
        self.ac1_2 = LeakyReLU(alpha=0.1)
        self.c1_3 = Conv2D(filters=512, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn1_3 = BatchNormalization()
        self.ac1_3 = LeakyReLU(alpha=0.1)
        self.c1_4 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn1_4 = BatchNormalization()
        self.ac1_4 = LeakyReLU(alpha=0.1)
        self.c1_5 = Conv2D(filters=512, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn1_5 = BatchNormalization()
        self.ac1_5 = LeakyReLU(alpha=0.1)

        # Predict one
        self.conv1_obj_branch = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn1_obj_branch = BatchNormalization()
        self.ac1_obj_branch = LeakyReLU(alpha=0.1)
        self.conv1_bbox = Conv2D(filters=3 * (self.num_class + 5), kernel_size=(1, 1), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.))

        self.conv_down1 = Conv2D(input_shape=(13, 13, 512), filters=256, kernel_size=(1, 1), padding='same',
                                 use_bias=False)
        self.bn_down1 = BatchNormalization()
        self.ac_down1 = LeakyReLU(alpha=0.1)

        # Convolution Set2
        self.c2_1 = Conv2D(input_shape=(26, 26, 768), filters=256, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn2_1 = BatchNormalization()
        self.ac2_1 = LeakyReLU(alpha=0.1)
        self.c2_2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn2_2 = BatchNormalization()
        self.ac2_2 = LeakyReLU(alpha=0.1)
        self.c2_3 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn2_3 = BatchNormalization()
        self.ac2_3 = LeakyReLU(alpha=0.1)
        self.c2_4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn2_4 = BatchNormalization()
        self.ac2_4 = LeakyReLU(alpha=0.1)
        self.c2_5 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn2_5 = BatchNormalization()
        self.ac2_5 = LeakyReLU(alpha=0.1)
        # Predict two
        self.conv2_obj_branch = Conv2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn2_obj_branch = BatchNormalization()
        self.ac2_obj_branch = LeakyReLU(alpha=0.1)
        self.conv2_bbox = Conv2D(filters=3 * (self.num_class + 5), kernel_size=(1, 1), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.))

        self.conv_down2 = Conv2D(input_shape=(26, 26, 256), filters=128, kernel_size=(1, 1), padding='same',
                                 use_bias=False)
        self.bn_down2 = BatchNormalization()
        self.ac_down2 = LeakyReLU(alpha=0.1)
        # Convolution Set3
        self.c3_1 = Conv2D(input_shape=(52, 52, 348), filters=128, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn3_1 = BatchNormalization()
        self.ac3_1 = LeakyReLU(alpha=0.1)
        self.c3_2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn3_2 = BatchNormalization()
        self.ac3_2 = LeakyReLU(alpha=0.1)
        self.c3_3 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn3_3 = BatchNormalization()
        self.ac3_3 = LeakyReLU(alpha=0.1)
        self.c3_4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn3_4 = BatchNormalization()
        self.ac3_4 = LeakyReLU(alpha=0.1)
        self.c3_5 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn3_5 = BatchNormalization()
        self.ac3_5 = LeakyReLU(alpha=0.1)
        # Predict three
        self.conv3_obj_branch = Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn3_obj_branch = BatchNormalization()
        self.ac3_obj_branch = LeakyReLU(alpha=0.1)
        self.conv3_bbox = Conv2D(filters=3 * (self.num_class + 5), kernel_size=(1, 1), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.))

    def call(self, x, training=True):
        self.train = training
        route_1, route_2, output = self.Darknet(x)
        # Convolution Set1
        output = self.c1_1(output)
        output = self.bn1_1(output)
        output = self.ac1_1(output)
        output = self.c1_2(output)
        output = self.bn1_2(output)
        output = self.ac1_2(output)
        output = self.c1_3(output)
        output = self.bn1_3(output)
        output = self.ac1_3(output)
        output = self.c1_4(output)
        output = self.bn1_4(output)
        output = self.ac1_4(output)
        output = self.c1_5(output)
        output = self.bn1_5(output)
        output = self.ac1_5(output)

        bbox1 = self.conv1_obj_branch(output)
        bbox1 = self.bn1_obj_branch(bbox1)
        bbox1 = self.ac1_obj_branch(bbox1)
        bbox1 = self.conv1_bbox(bbox1)

        output = self.conv_down1(output)
        output = self.bn_down1(output)
        output = self.ac_down1(output)
        output = utils.upsample(output)
        output = tf.concat([output, route_2], axis=-1)

        # Convolution Set2
        output = self.c2_1(output)
        output = self.bn2_1(output)
        output = self.ac2_1(output)
        output = self.c2_2(output)
        output = self.bn2_2(output)
        output = self.ac2_2(output)
        output = self.c2_3(output)
        output = self.bn2_3(output)
        output = self.ac2_3(output)
        output = self.c2_4(output)
        output = self.bn2_4(output)
        output = self.ac2_4(output)
        output = self.c2_5(output)
        output = self.bn2_5(output)
        output = self.ac2_5(output)

        bbox2 = self.conv2_obj_branch(output)
        bbox2 = self.bn2_obj_branch(bbox2)
        bbox2 = self.ac2_obj_branch(bbox2)
        bbox2 = self.conv2_bbox(bbox2)

        output = self.conv_down2(output)
        output = self.bn_down2(output)
        output = self.ac_down2(output)
        output = utils.upsample(output)
        output = tf.concat([output, route_1], axis=-1)

        # Convolution Set3
        output = self.c3_1(output)
        output = self.bn3_1(output)
        output = self.ac3_1(output)
        output = self.c3_2(output)
        output = self.bn3_2(output)
        output = self.ac3_2(output)
        output = self.c3_3(output)
        output = self.bn3_3(output)
        output = self.ac3_3(output)
        output = self.c3_4(output)
        output = self.bn3_4(output)
        output = self.ac3_4(output)
        output = self.c3_5(output)
        output = self.bn3_5(output)
        output = self.ac3_5(output)

        bbox3 = self.conv3_obj_branch(output)
        bbox3 = self.bn3_obj_branch(bbox3)
        bbox3 = self.ac3_obj_branch(bbox3)
        bbox3 = self.conv3_bbox(bbox3)
        conv_tensors = [bbox3, bbox2, bbox1]
        output_tensors = []
        if self.train == True:
            for i, conv_tensor in enumerate(conv_tensors):
                pred_tensor = self.decode(conv_tensor, i)
                output_tensors.append(conv_tensor)
                output_tensors.append(pred_tensor)
            return output_tensors
        else:
            for i, conv_tensor in enumerate(conv_tensors):
                pred_tensor = self.decode(conv_tensor, i)
                output_tensors.append(pred_tensor)
            return output_tensors

    def decode(self, conv_output, i=0):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
                contains (x, y, w, h, score, probability)
        """
        num_class = len(utils.read_class_names(self.opt.data))
        strides = np.array([8, 16, 32])
        anchors = np.array(utils.get_anchors(self.opt.data))

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(opt, pred, conv, label, bboxes, i=0):
    num_class = len(utils.read_class_names(opt.data))
    strides = np.array([8, 16, 32])
    IOU_LOSS_THRESH = opt.iou_loss_thresh
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = strides[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_class))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
