import tensorflow as tf


def get_data_augmenter(
        adaption_dataset: tf.data.Dataset = None,
        resize_image_size: int = 72,
        flip_mode: str = 'horizontal_and_vertical',
        rotation_factor: float = 0.02,
        zoom_height_factor: float = 0.2,
        zoom_width_factor: float = 0.2,
):
    """
    :param adaption_dataset:
    :param resize_image_size:
    :param flip_mode:
    :param rotation_factor:
    :param zoom_height_factor:
    :param zoom_width_factor:
    :return:
    """
    data_augmentation = tf.keras.Sequential(
        [
            # tf.keras.layers.Normalization(),
            tf.keras.layers.Resizing(resize_image_size, resize_image_size),
            tf.keras.layers.RandomFlip(flip_mode),
            tf.keras.layers.RandomRotation(factor=rotation_factor),
            tf.keras.layers.RandomZoom(
                height_factor=zoom_height_factor, width_factor=zoom_width_factor
            ),
        ],
        name="data_augmentation",
    )

    # if adaption_dataset is not None:
    #     data_augmentation.layers[0].adapt(adaption_dataset)
    return data_augmentation
