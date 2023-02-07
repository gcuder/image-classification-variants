from keras import applications
import tensorflow as tf

from computer_vision import registry
from computer_vision.lab.models.backbones.base import Backbone


class EfficientNet(Backbone):
    def __init__(self, weights: str, trainable: bool = True, **kwargs):
        super(EfficientNet, self).__init__(trainable=trainable, **kwargs)
        self._pooling = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")
        self._bn = tf.keras.layers.BatchNormalization()
        self._weights = weights

    def input_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """
        return inputs

    def output_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """
        x = self._bn(inputs)
        x = self._pooling(x)
        return x


@registry.backbone_registry.register('efficient_net_b0')
class EfficientNetB0(EfficientNet):
    """ TODO
    """

    def __init__(self, weights: str, trainable: bool = True, **kwargs):
        self._backbone = applications.EfficientNetB0(include_top=False, weights=weights)
        super(EfficientNetB0, self).__init__(weights=weights,
                                             trainable=trainable,
                                             **kwargs)


@registry.backbone_registry.register('efficient_net_b1')
class EfficientNetB1(EfficientNet):
    """ TODO
    """

    def __init__(self, weights: str, trainable: bool = True, **kwargs):
        self._backbone = applications.EfficientNetB1(include_top=False, weights=weights)
        super(EfficientNetB1, self).__init__(weights=weights,
                                             trainable=trainable,
                                             **kwargs)


__all__ = ['registry']
