from absl import app, flags, logging  # noqa: E402

from computer_vision.lab.augmentation import get_data_augmenter
from computer_vision.lab.data import build_dataset
from computer_vision.lab.models import baseline

IMAGE_SIZE = (512, 512)


def main():
    FLAGS = flags.FLAGS

    dataset_kwargs = dict(
        dataset_card=FLAGS.dataset_card,
        batch_size=FLAGS.batch_size,
        data_dir=FLAGS.data_dir,
        image_size=IMAGE_SIZE,
        shuffle_buffer=FLAGS.shuffle_buffeer
    )
    train_ds, info = build_dataset(split='train', **dataset_kwargs)
    dev_ds, _ = build_dataset(split='validation', **dataset_kwargs)
    test_ds, _ = build_dataset(split='test', **dataset_kwargs)

    config = baseline.ClassifierWithBackboneConfig(num_classes=info.features['label'].num_classes)
    # num_classes from info
    augmentation = get_data_augmenter()
    model = baseline.ClassifierWithBackbone(config=config, augmentation=augmentation)
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
    pass


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_flags()
    app.run(main)
