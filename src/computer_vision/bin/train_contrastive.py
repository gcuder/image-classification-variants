from absl import app, flags, logging  # noqa: E402
import tensorflow as tf
import tensorflow_addons as tfa

from computer_vision.lab.augmentation import get_data_augmenter
from computer_vision.lab.data import build_dataset
from computer_vision.lab.models import baseline
from computer_vision.lab.models.backbones import BACKBONES, backbones
from computer_vision.lab.utils import unfreeze_model

IMAGE_SIZE = (224, 224)
input_shape = (224, 224, 3)


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def main(_):
    FLAGS = flags.FLAGS
    learning_rate = 0.001
    batch_size = 265
    hidden_units = 512
    projection_units = 128
    num_epochs = 50
    dropout_rate = 0.5
    temperature = 0.05

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

    augmentation = get_data_augmenter()

    def create_encoder():
        backbone = backbones(name=FLAGS.backbone,
                             variant='EfficientNetV2S',
                             weights='imagenet',
                             trainable=True,
                             pooling='avg')

        inputs = tf.keras.Input(shape=input_shape)
        augmented = augmentation(inputs)
        outputs = backbone(augmented)
        _model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet-encoder")
        return _model

    encoder_model = create_encoder()

    def add_projection_head(encoder):
        inputs = tf.keras.Input(shape=input_shape)
        features = encoder(inputs)
        outputs = tf.keras.layers.Dense(projection_units, activation="relu")(features)
        _model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
        )
        return _model

    encoder_with_ph = add_projection_head(encoder_model)

    encoder_with_ph.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=SupervisedContrastiveLoss(temperature),
    )

    encoder_with_ph.summary()

    history = encoder_with_ph.fit(train_ds, validation_data=dev_ds, batch_size=batch_size, epochs=num_epochs,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5)])

    # Classfication model part
    trained_backbone = encoder_with_ph.layers[1].layers[2]
    trained_backbone.trainable = False

    config = baseline.ClassifierWithBackboneConfig(num_classes=info.features['label'].num_classes)
    model = baseline.ClassifierWithBackbone(config=config,
                                            augmentation=augmentation,
                                            backbone=trained_backbone)
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
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics
    )

    epochs = 40
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


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_flags()
    app.run(main)
