from typing import *

import tensorflow_datasets as tfds
import tensorflow as tf


def _get_datasets(dataset_cards: str,
                  train_dev_split: float = 0.1,
                  data_dir: str = None,
                  seed: int = 123
                  ) -> Tuple[Mapping[Text, tf.data.Dataset], tfds.core.DatasetInfo]:
    datasets, info = tfds.load(dataset_cards,
                               data_dir=data_dir,
                               as_supervised=True,
                               with_info=True)
    train_ds: tf.data.Dataset = datasets['train'].shuffle(256, seed=seed)
    test_ds = datasets['test']
    if 'validation' not in datasets.keys():
        num_samples = int(train_ds.cardinality())
        dev_samples = int(num_samples*train_dev_split)
        # train_ds, dev_ds = tf.keras.utils.split_dataset(dataset=train_ds,
        #                                                 left_size=1 - train_dev_split,
        #                                                 right_size=train_dev_split)
        dev_ds = train_ds.take(dev_samples)
        train_ds = train_ds.skip(dev_samples)
    else:
        dev_ds = datasets['validation']

    return {
               'train': train_ds,
               'validation': dev_ds,
               'test': test_ds
           }, info


def build_dataset(dataset_cards: str,
                  image_size: Tuple[int, int],
                  train_dev_split: float = 0.1,
                  data_dir: str = None,
                  batch_size: int = 256,
                  shuffle_buffer: Optional[int] = None,
                  repeat: bool = False,
                  seed: int = 123) -> tf.data.Dataset:
    splits, info = _get_datasets(dataset_cards=dataset_cards,
                                 data_dir=data_dir,
                                 train_dev_split=train_dev_split,
                                 seed=seed)

    NUM_CLASSES = info.features["label"].num_classes
    for split, dataset in splits.items():
        is_train = split == "train"
        map_kwargs = dict(num_parallel_calls=tf.data.AUTOTUNE)

        if is_train and shuffle_buffer is not None:
            dataset = dataset.shuffle(shuffle_buffer)

        if is_train and repeat:
            dataset = dataset.repeat()

        # Resize the images
        dataset = dataset.map(lambda image, label: (tf.image.resize(image, image_size), tf.one_hot(label, NUM_CLASSES)))
        dataset = dataset.batch(batch_size, drop_remainder=not is_train, **map_kwargs)
        splits.update({split: dataset})
    return splits, info


# if __name__ == '__main__':
#     datasets, info = build_dataset(dataset_cards=image_size=(512, 512))
#
#     print('all done')
