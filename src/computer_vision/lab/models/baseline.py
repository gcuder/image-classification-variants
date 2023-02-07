from typing import List

import tensorflow as tf
from computer_vision.lab.models.base import Model, ModelConfig, ModelWithBackbone
from computer_vision.lab.layers import ConvolutionalBlock


class BaselineClassifierConfig(ModelConfig):
    num_layers: int = 3
    filters: List[int] = [32, 64, 128]


class BaselineClassifier(Model):
    def __init__(self, config: BaselineClassifierConfig):
        super(BaselineClassifier, self).__init__(config=config)

        self._num_layers = config.num_layers
        self._filters = config.filters

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

    def _body(self, inputs, training=None) -> tf.Tensor:
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        return x


class EfficientNetClassifierConfig(ModelConfig):
    pass


class EfficientNetClassifier(ModelWithBackbone):
    def __init__(self, config: EfficientNetClassifierConfig):
        super(EfficientNetClassifier, self).__init__(config=config)


if __name__ == '__main__':
    model = BaselineClassifier(config=BaselineClassifierConfig(num_classes=10))
    model.build()
    model.summary()
