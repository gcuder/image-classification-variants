from absl import app, flags, logging  # noqa: E402

from computer_vision.lab.augmentation import get_data_augmenter
from computer_vision.lab.data import build_dataset
from computer_vision.lab.models import baseline

IMAGE_SIZE = (512, 512)

from computer_vision.lab.models.backbones import EfficientNet
def main(_):
    FLAGS = flags.FLAGS

    dataset_kwargs = dict(
        dataset_cards=FLAGS.dataset_cards,
        batch_size=FLAGS.batch_size,
        data_dir=FLAGS.data_dir,
        image_size=IMAGE_SIZE,
        shuffle_buffer=FLAGS.shuffle_buffer
    )
    train_ds, info = build_dataset(split='train', **dataset_kwargs)
    dev_ds, _ = build_dataset(split='validation', **dataset_kwargs)
    test_ds, _ = build_dataset(split='test', **dataset_kwargs)

    config = baseline.ClassifierWithBackboneConfig(num_classes=info.features['label'].num_classes)
    augmentation = get_data_augmenter()
    model = baseline.ClassifierWithBackbone(config=config,
                                            augmentation=augmentation,
                                            backbone=EfficientNet(variant='EfficientNetV2S',
                                                                  weights=None))
    model.build()
    model.summary()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    epochs = 10
    hist = model.fit(train_ds, epochs=epochs, validation_data=dev_ds)
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
