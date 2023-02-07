import abc
from typing import Any, Mapping, Union

import tensorflow as tf


class Backbone(tf.keras.layers.Layer, abc.ABC):
    """ TODO
    """
    _backbone: Union[tf.keras.layers.Layer, tf.keras.Model]

    def __init__(self, trainable: bool = True, **kwargs):
        super(Backbone, self).__init__(**kwargs)
        self._trainable = trainable
        self._backbone.trainable = trainable

    def call(self, inputs, *args, **kwargs):
        """ TODO
        """
        x = self.input_processor(inputs=inputs)
        x = self._backbone(x)
        x = self.output_processor(inputs=x)
        return x

    @abc.abstractmethod
    def input_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """

    @abc.abstractmethod
    def output_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """

    def get_config(self) -> Mapping[str, Any]:
        """ TODO
        """
        config = super(Backbone, self).get_config()
        _config = {
            'trainable': self._trainable
        }
        config.update(_config)
        return config
