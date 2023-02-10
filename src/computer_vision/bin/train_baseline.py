from absl import app, flags, logging  # noqa: E402
import tensorflow as tf
import tensorflow_addons as tfa

from computer_vision.lab.augmentation import get_data_augmenter
from computer_vision.lab.data import build_dataset
from computer_vision.lab.models import baseline
from computer_vision.lab.models.backbones import BACKBONES, backbones
from computer_vision.lab.utils import unfreeze_model

IMAGE_SIZE = (224, 224)


def main(_):
    FLAGS = flags.FLAGS

    dataset_kwargs = dict(
        train_dev_split=0.25,
        dataset_cards=FLAGS.dataset_cards,
        batch_size=FLAGS.batch_size,
        data_dir=FLAGS.data_dir,
        image_size=IMAGE_SIZE,
        shuffle_buffer=FLAGS.shuffle_buffer,
        one_hot_labels=True
    )
    datasets, info = build_dataset(**dataset_kwargs)

    train_ds = datasets['train']
    dev_ds = datasets['validation']
    test_ds = datasets['test']

    config = baseline.ClassifierWithBackboneConfig(num_classes=info.features['label'].num_classes)
    augmentation = get_data_augmenter()
    backbone = FLAGS.backbone
    if backbone is not None:
        backbone = backbones(name=backbone, variant='EfficientNetV2S',
                             weights='imagenet',
                             trainable=FLAGS.unfreeze_backbone,
                             pooling='avg')
    # backbone.unfreeze_model(n=20, from_top=False)

    model = baseline.ClassifierWithBackbone(config=config,
                                            augmentation=augmentation,
                                            backbone=backbone)
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics
    )

    epochs = 40
    hist = model.fit(train_ds, epochs=epochs, validation_data=dev_ds, callbacks=callbacks)
    results = model.evaluate(test_ds, return_dict=True)
    print(results)


def define_flags():
    flags.DEFINE_string(
        name="model_dir", required=False, default=None, help="Directory, where all model results are saved!"
    )

    flags.DEFINE_string(
        name="data_dir", default=None, help="Directory, where the training data is stored"
    )

    flags.DEFINE_enum(
        name="backbone", default=None, enum_values=list(BACKBONES.keys()),
        help=""
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

    flags.DEFINE_bool(
        name="unfreeze_backbone",
        default=False,
        help=""
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_flags()
    app.run(main)
