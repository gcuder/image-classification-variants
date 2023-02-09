import tensorflow as tf


def get_training_strategy(num_gpus: int):
    pass


def make_model_dir(model_root: str):
    pass


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[0].layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    return model
