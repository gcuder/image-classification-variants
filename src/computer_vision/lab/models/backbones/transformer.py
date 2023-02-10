from typing import Literal, Optional

from keras import applications
import tensorflow as tf
from transformers import AutoImageProcessor, TFAutoModel, shape_list

from computer_vision.lab.models.backbones.base import Backbone


# def _get_efficient_net(name: str, trainable: bool, **kwargs):
#     try:
#         net = EFFICIENT_NETS[name](**kwargs)
#         net.trainable = trainable
#         return net
#     except KeyError:
#         raise ValueError("Variant '%s' not found." % name)

def _get_hf_layer(checkpoint: str, trainable: bool = True, **kwargs) -> tf.keras.layers.Layer:
    """Extracts the core layer of a TFAutoModel that reflects the embedding.

    :param checkpoint: String defining the checkpoint
    :param trainable: Whether weights should be updateable or not.
    :param kwargs: Additional kwargs
    :return: A pretrained layer of type tf.keras.layers.Layer
    """
    main_layer = TFAutoModel.from_pretrained(checkpoint, **kwargs).layers[0]
    main_layer.trainable = trainable
    return main_layer


class VisionTransformer(Backbone):
    """TODO
    """

    def __init__(self,
                 checkpoint: str,
                 trainable: bool = True,
                 average_pixel: bool = False,
                 **kwargs):
        super(VisionTransformer, self).__init__(trainable=trainable, **kwargs)
        self._checkpoint = checkpoint
        self._average_pixel = average_pixel
        self._backbone = _get_hf_layer(checkpoint=checkpoint, trainable=trainable)
        self._layer_norm = tf.keras.layers.LayerNormalization(name='cls_layer_norm')

    def input_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """
        # Transpose inputs fom [B, H, W, C] to [B, C, H, W]
        inputs_t = tf.transpose(inputs, perm=[0, 3, 1, 2])
        return inputs_t

    def output_processor(self, inputs: tf.Tensor) -> tf.Tensor:
        """ TODO
        """
        x = inputs
        sequence_output = x[0]
        cls_token = x[1]
        if not self._average_pixel:
            sequence_output = self._layer_norm(cls_token)
        else:
            # rearrange "batch_size, num_channels, height, width -> batch_size, (height*width), num_channels"
            batch_size, num_channels, height, width = shape_list(sequence_output)
            sequence_output = tf.reshape(sequence_output, shape=(batch_size, num_channels, height * width))
            sequence_output = tf.transpose(sequence_output, perm=(0, 2, 1))
            sequence_output = self._layer_norm(sequence_output)
        sequence_output_mean = tf.reduce_mean(sequence_output, axis=1)
        return sequence_output_mean


if __name__ == '__main__':
    vit = VisionTransformer(checkpoint='microsoft/cvt-13', trainable=False, return_sequences=True)
    inputs = tf.random.uniform(shape=(1, 224, 224, 3))
    outputs = vit(inputs)
    print(outputs)
