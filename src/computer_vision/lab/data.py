from typing import Optional, Tuple

import tensorflow_datasets as tfds
import tensorflow as tf


def build_dataset(split: str,
                  image_size: Tuple[int, int],
                  data_dir: str = None,
                  dataset_cards: str = 'oxford_flowers102',
                  batch_size: int = 256,
                  shuffle_buffer: Optional[int] = None,
                  repeat: bool = False) -> tf.data.Dataset:
    is_train = split == "train"
    map_kwargs = dict(num_parallel_calls=tf.data.AUTOTUNE)
    _, info = tfds.load(dataset_cards,
                        split=split,
                        data_dir=data_dir,
                        with_info=True)
    dataset = tfds.load(dataset_cards,
                        split=split,
                        data_dir=data_dir,
                        as_supervised=True)

    if is_train and shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)

    if is_train and repeat:
        dataset = dataset.repeat()

    # Resize the images
    dataset = dataset.map(lambda image, label: (tf.image.resize(image, image_size), label))
    dataset = dataset.batch(batch_size, drop_remainder=not is_train, **map_kwargs)
    return dataset, info


if __name__ == '__main__':
    train_ds, info = build_dataset(split='train', image_size=(512, 512))

    print('all done')
