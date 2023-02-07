
from computer_vision.lab.augmentation import get_data_augmenter
from computer_vision.lab.data import build_dataset
from computer_vision.lab.models import baseline
IMAGE_SIZE = (512, 512)


def main():
    train_ds, info = build_dataset(split='train', image_size=IMAGE_SIZE, batch_size=16)
    dev_ds, _ = build_dataset(split='validation', image_size=IMAGE_SIZE, batch_size=16)
    test_ds, _ = build_dataset(split='test', image_size=IMAGE_SIZE, batch_size=16)

    config = baseline.ClassifierWithBackboneConfig(num_classes=102)
    augmentation = get_data_augmenter()
    model = baseline.ClassifierWithBackbone(config=config, augmentation=augmentation)
    model.build()
    model.summary()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    epochs = 10  # @param {type: "slider", min:10, max:100}
    hist = model.fit(train_ds, epochs=epochs, validation_data=dev_ds)
    results = model.evaluate(test_ds, return_dict=True)
    print(results)





if __name__ == '__main__':
    main()
