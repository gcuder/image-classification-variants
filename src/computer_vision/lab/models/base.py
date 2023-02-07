from typing import Union

import tensorflow as tf
from pydantic import BaseModel
import abc


class ModelConfig(BaseModel):
    num_classes: int
    rescaling_factor: float = 1.0 / 255
    output_dropout_rate: float = 0.5


class Model(tf.keras.Model, abc.ABC):
    """ TODO
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Model, self).__init__(**kwargs)
        self._num_classes = config.num_classes
        self._rescaling_factor = config.rescaling_factor
        self._output_dropout_rate = config.output_dropout_rate

        if self._num_classes == 2:
            output_activation = "sigmoid"
            output_units = 1
        else:
            output_activation = "softmax"
            output_units = self._num_classes

        self._cls_dropout = tf.keras.layers.Dropout(self._output_dropout_rate)
        self._cls_head = tf.keras.layers.Dense(activation=output_activation, units=output_units)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = inputs
        x = self._input_processor(inputs=x, training=training)
        x = self._body(inputs=x, training=training)
        x = self._top(inputs=x, training=training)
        return x

    def _input_processor(self, inputs, training=None):
        x = tf.keras.layers.Rescaling(self._rescaling_factor)(inputs)
        return x

    @abc.abstractmethod
    def _body(self, inputs, training=None) -> tf.Tensor:

        """
        :param inputs:
        :return:
        """

    def _top(self, inputs, training=None):
        x = self._cls_dropout(inputs, training=training)
        x = self._cls_head(x)
        return x

    def build(self, input_shape=None):
        if input_shape is not None:
            super(Model, self).build(input_shape=input_shape)

        if self.built:
            return

        self.built = True

        input = tf.random.uniform(shape=(2, 512, 512, 3))
        # noinspection PyCallingNonCallable
        self(inputs=input)


class ModelWithBackbone(Model):
    """ TODO

    """
    _backbone: Union[tf.keras.layers.Layer, tf.keras.Model]

    def __init__(self, config: ModelConfig):
        super(ModelWithBackbone, self).__init__(config=config)

    def _body(self, inputs, training=None) -> tf.Tensor:
        return self._backbone(inputs)
