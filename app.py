import math

import tensorflow as tf
import os

import helpers
import settings

AUTOTUNE = tf.data.experimental.AUTOTUNE

assert tf.test.is_built_with_cuda()
assert tf.test.is_gpu_available()

# Collect the list of training files and process their paths.
training_dataset_files = tf.data.Dataset.list_files(os.path.join(settings.TRAINING_DIRECTORY, '*', '*.png'))
training_dataset_labelled = training_dataset_files.map(helpers.process_path, num_parallel_calls=AUTOTUNE)
training_dataset = helpers.prepare_for_training(training_dataset_labelled)

# Collect the validation files.
validation_dataset_files = tf.data.Dataset.list_files(os.path.join(settings.VALIDATION_DIRECTORY, '*', '*.png'))
validation_dataset_labelled = validation_dataset_files.map(helpers.process_path, num_parallel_calls=AUTOTUNE)
validation_dataset = helpers.prepare_for_training(validation_dataset_labelled)

# image_batch, label_batch = next(iter(training_dataset))
# helpers.show_batch(image_batch.numpy(), label_batch.numpy())

# image_batch, label_batch = next(iter(validation_dataset))
# helpers.show_batch(image_batch.numpy(), label_batch.numpy())

use_latest_model_file = False
if use_latest_model_file:
    model = tf.keras.models.load_model(settings.MODEL_FILE)
else:
    model = tf.keras.models.Sequential([
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(settings.TARGET_IMAGE_HEIGHT, settings.TARGET_IMAGE_WIDTH, 3)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),

        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),

        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('false') and 1 for the other ('true')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['acc']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        patience=2,
        verbose=1,
        factor=0.5,
        min_lr=0.00001,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        filepath=settings.CHECKPOINT_FILE,
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1,
    ),
    # tf.keras.callbacks.TensorBoard(
    #     log_dir=settings.LOG_DIRECTORY,
    #     histogram_freq=1
    # )
]

training_dataset_length = tf.data.experimental.cardinality(training_dataset_files).numpy()
steps_per_epoch = math.ceil(training_dataset_length // settings.TRAINING_BATCH_SIZE)
print(f'Training dataset length: {training_dataset_length}, batch size: {settings.TRAINING_BATCH_SIZE}, steps_per_epoch: {steps_per_epoch}')

validation_dataset_length = tf.data.experimental.cardinality(validation_dataset_files).numpy()
validation_steps = math.ceil(validation_dataset_length // settings.VALIDATION_BATCH_SIZE)
print(f'Validation dataset length: {validation_dataset_length}, batch size: {settings.VALIDATION_BATCH_SIZE}, validation_steps: {validation_steps}')

history = model.fit_generator(
    training_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=30,  # 20000,
    verbose=2 if 'PYCHARM_HOSTED' in os.environ else 1,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks,
    # use_multiprocessing=True,
    # workers=8,
)

model.save(settings.FULL_MODEL_FILE)
