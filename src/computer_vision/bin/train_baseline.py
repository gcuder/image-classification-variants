from absl import app, flags, logging  # noqa: E402
import tensorflow as tf
import tensorflow_addons as tfa

from computer_vision.lab.augmentation import get_data_augmenter
from computer_vision.lab.data import build_dataset
from computer_vision.lab.models import baseline

IMAGE_SIZE = (224, 224)

from computer_vision.lab.models.backbones import EfficientNet


def main(_):
    FLAGS = flags.FLAGS

    dataset_kwargs = dict(
        train_dev_split=0.25,
        dataset_cards=FLAGS.dataset_cards,
        batch_size=FLAGS.batch_size,
        data_dir=FLAGS.data_dir,
        image_size=IMAGE_SIZE,
        shuffle_buffer=FLAGS.shuffle_buffer
    )
    datasets, info = build_dataset(**dataset_kwargs)

    train_ds = datasets['train']
    dev_ds = datasets['validation']
    test_ds = datasets['test']

    config = baseline.ClassifierWithBackboneConfig(num_classes=info.features['label'].num_classes)
    augmentation = get_data_augmenter()
    model = baseline.ClassifierWithBackbone(config=config,
                                            augmentation=augmentation,
                                            backbone=EfficientNet(variant='EfficientNetV2S',
                                                                  weights='imagenet',
                                                                  trainable=False,
                                                                  pooling='avg'))
    model.build()
    model.summary()

    metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tfa.metrics.F1Score(num_classes=config.num_classes, average='micro', name='micro_f1'),
               tfa.metrics.F1Score(num_classes=config.num_classes, average='macro', name='macro_f1'),
               tfa.metrics.F1Score(num_classes=config.num_classes, average='weighted', name='weighted_f1'),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=2)]
    model.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )

    epochs = 40
    hist = model.fit(train_ds, epochs=epochs, validation_data=dev_ds, callbacks=callbacks)
    results = model.evaluate(test_ds, return_dict=True)
    print(results)


def define_flags():
    flags.DEFINE_string(
        name="model_root", required=False, default=None, help="Directory, where all model results are saved!"
    )

    flags.DEFINE_string(
        name="data_dir", default=None, help="Directory, where the training data is stored"
    )

    flags.DEFINE_string(
        name="config", required=False, default=None, help="Directory, where all model results are saved!"
    )

    flags.DEFINE_string(
        name="dataset_cards", default="oxford_flowers102", help=""
    )

    flags.DEFINE_integer(
        name="batch_size",
        default=64,
        help=""
    )

    flags.DEFINE_integer(
        name="shuffle_buffer",
        default=256,
        help=""
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_flags()
    app.run(main)
