from typing import Tuple

import tensorflow as tf


class ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Tuple[int, int], pool_size: Tuple[int, int], padding: str =
    'valid', **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)
        self._filters = filters

        self._activation = tf.keras.layers.Activation('relu')
        self._bn = tf.keras.layers.BatchNormalization()
        self._conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding)
        self._max_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)

    def call(self, inputs, *args, **kwargs):
        x = self._conv(inputs)
        x = self._activation(x)
        x = self._bn(x)
        x = self._max_pool(x)
        return x


class SeparableConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super(SeparableConvolutionalBlock, self).__init__(**kwargs)
        self._filters = filters

        self._activation = tf.keras.layers.Activation('relu')
        self._bn_1 = tf.keras.layers.BatchNormalization()
        self._sep_conv_1 = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")

    def call(self, inputs, *args, **kwargs):
        x = self._activation(inputs)
        x = self._sep_conv_1(x)
        x = self._bn_1(x)
        return x
#
#
# class XceptionLayer(tf.keras.layers.Layer):
#     def __init__(self, filters: int, **kwargs):
#         super(XceptionLayer, self).__init__(**kwargs)
#         self._filters = filters
#
#         self._conv_1 = ConvolutionalBlock(filters=filters)
#         self._conv_2 = ConvolutionalBlock(filters=filters)
#         self._max_pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")
#
#         self._res_conv = tf.keras.layers.Conv2D(filters=filters, 1, strides=2, padding="same")
#
#     def call(self, inputs, *args, **kwargs):
#         """
#         :param inputs:
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         previous_block_activation = inputs
#         x = self._conv_1(inputs)
#         x = self._conv_2(x)
#         x = self._conv_3(x)
#
#         x = self._max_pool(x)
#
#         residual = self._res_conv(previous_block_activation)
#         x += residual
