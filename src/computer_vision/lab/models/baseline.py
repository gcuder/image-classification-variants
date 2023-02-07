from typing import Any, List, Mapping, Optional, Text

import tensorflow as tf
from computer_vision.lab.models.base import Model, ModelConfig, ModelWithBackbone
from computer_vision.lab.layers import ConvolutionalBlock
from computer_vision import registry


class BaselineClassifierConfig(ModelConfig):
    num_layers: int = 3
    rescaling_factor: float = 1.0 / 255
    filters: List[int] = [32, 64, 128]


class BaselineClassifier(Model):
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

    def _input_processor(self, inputs, training=None):
        return self._rescaling_layer(inputs)

    def _body(self, inputs, training=None) -> tf.Tensor:
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        return x


class ClassifierWithBackboneConfig(ModelConfig):
    backbone: str = 'efficient_net_b0'
    backbone_kwargs: Mapping[Text, Any] = {
        'trainable': True,
        'weights': None
    }


class ClassifierWithBackbone(ModelWithBackbone):
    def __init__(self, config: ClassifierWithBackboneConfig, augmentation: Optional[tf.keras.Model] = None, **kwargs):
        super(ClassifierWithBackbone, self).__init__(config=config, augmentation=augmentation, **kwargs)
        self._backbone = registry.backbones(config.backbone, **config.backbone_kwargs)


if __name__ == '__main__':
    model = ClassifierWithBackbone(config=ClassifierWithBackboneConfig(num_classes=10))
    model.build()
    model.summary()

    input = tf.random.uniform(shape=(2, 512, 512, 3))
    x = model(input)
    print(x)
