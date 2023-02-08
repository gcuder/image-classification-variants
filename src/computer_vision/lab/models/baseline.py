from typing import Any, List, Mapping, Optional, Text

import tensorflow as tf
from computer_vision.lab.models.base import ClassficationModelConfig, ClassificationModel
from computer_vision.lab.layers import ConvolutionalBlock


class BaselineClassifierConfig(ClassficationModelConfig):
    num_layers: int = 3
    rescaling_factor: float = 1.0 / 255
    filters: List[int] = [32, 64, 128]


class BaselineClassifier(ClassificationModel):
    def __init__(self, config: BaselineClassifierConfig):
        super(BaselineClassifier, self).__init__(config=config)

        self._num_layers = config.num_layers
        self._filters = config.filters

        self._rescaling_layer = tf.keras.layers.Rescaling(config.rescaling_factor)
        if len(self._filters) != self._num_layers:
            raise ValueError()

        self._conv_layers = []

        for i in range(self._num_layers):
            conv_block = ConvolutionalBlock(filters=self._filters[i],
                                            kernel_size=(3, 3),
                                            pool_size=(2, 2),
                                            name='conv_layer_%d' % i)
            self._conv_layers.append(conv_block)
        self._flatten = tf.keras.layers.Flatten()

    def _bottom(self, inputs, training=None):
        x = inputs
        x = self._rescaling_layer(x)
        x = super(BaselineClassifier, self)._bottom(x)
        return x

    def _body(self, inputs, training=None) -> tf.Tensor:
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        return x


class ClassifierWithBackboneConfig(ClassficationModelConfig):
    pass


class ClassifierWithBackbone(ClassificationModel):
    def __init__(self,
                 config: ClassifierWithBackboneConfig,
                 backbone: tf.keras.Model,
                 augmentation: Optional[tf.keras.Model] = None,
                 **kwargs):
        super(ClassifierWithBackbone, self).__init__(config=config,
                                                     augmentation=augmentation,
                                                     backbone=backbone,
                                                     **kwargs)

    def _body(self, inputs, training=None) -> tf.Tensor:
        return inputs

# if __name__ == '__main__':
#     model = ClassifierWithBackbone(config=ClassifierWithBackboneConfig(num_classes=10))
#     model.build()
#     model.summary()
#
#     input = tf.random.uniform(shape=(2, 512, 512, 3))
#     x = model(input)
#     print(x)
