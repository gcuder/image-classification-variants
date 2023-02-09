from typing import Literal, Optional

from keras import applications
import tensorflow as tf

from computer_vision.lab.models.backbones.base import Backbone

POOLING = Literal['avg', 'max']

EFFICIENT_NETS = {
    applications.EfficientNetV2B0.__name__: applications.EfficientNetV2B0,
    applications.EfficientNetV2B1.__name__: applications.EfficientNetV2B1,
    applications.EfficientNetV2B2.__name__: applications.EfficientNetV2B2,
    applications.EfficientNetV2B3.__name__: applications.EfficientNetV2B3,
    applications.EfficientNetV2S.__name__: applications.EfficientNetV2S,
    applications.EfficientNetV2M.__name__: applications.EfficientNetV2M,
    applications.EfficientNetV2L.__name__: applications.EfficientNetV2L,
}


def _get_efficient_net(name: str, trainable: bool, **kwargs):
    try:
        net = EFFICIENT_NETS[name](**kwargs)
        net.trainable = trainable
        return net
    except KeyError:
        raise ValueError("Variant '%s' not found." % name)


class EfficientNet(Backbone):
    """TODO
    """

    def __init__(self,
                 variant: str,
                 weights: str,
                 trainable: bool = True,
                 pooling: Optional[POOLING] = None,
                 **kwargs):
        super(EfficientNet, self).__init__(trainable=trainable, **kwargs)
        self._variant = variant
        self._weights = weights
        self._pooling = pooling
        self._backbone = _get_efficient_net(name=variant,
                                            trainable=trainable,
                                            weights=weights,
                                            pooling=pooling,
                                            include_top=False)

        self._n_layers = len(self._backbone.layers)

    def input_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """
        return inputs

    def output_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """
        return inputs

    def unfreeze_model(self, n: int, from_top: bool = False):
        self.trainable = True
        if n > self._n_layers:
            n = self._n_layers

        r = range(-1, -n) if from_top else range(0, n)
        for i in r:
            if not isinstance(self._backbone.layers[i], tf.keras.layers.BatchNormalization):
                self._backbone.layers[i].trainable = True
