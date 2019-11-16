import tensorflow as tf
import settings
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
"""


def random_bool(true_percentage: float = 1):
    # Default to a 20% chance
    weights = [true_percentage, 1 - true_percentage]
    return random.choices([True, False], weights=weights, k=1)[0]


def augment_image(img):
    # Apply transformations to the given image in order to multiply the number of images in the dataset.

    if random_bool():
        img = tf.image.random_crop(img, [
            int(settings.SOURCE_IMAGE_HEIGHT // 1.25),
            int(settings.SOURCE_IMAGE_WIDTH // 1.25),
            3,
        ])

    if random_bool():
        img = tf.image.flip_left_right(img)

    if random_bool():
        img = tf.image.random_hue(img, 0.1)

    if random_bool():
        img = tf.image.random_contrast(img, 0.5, 1.5)

    if random_bool():
        img = tf.image.random_saturation(img, 0.5, 1.5)

    if random_bool():
        img = tf.image.random_brightness(img, 0.125)

    img = tf.clip_by_value(img, 0.0, 1.0)
    # img = tf.math.maximum(img, 1.0)
    # img = tf.math.minimum(img, 0.0)

    return img


def process_path(file_path):
    parts = tf.strings.split(file_path, '\\')
    label = parts[-2] == settings.CLASS_NAMES

    # Read the file and decode the image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = augment_image(img)

    img = tf.image.resize(img, [settings.TARGET_IMAGE_HEIGHT, settings.TARGET_IMAGE_WIDTH])
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=10000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()
    ds = ds.batch(settings.TRAINING_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def show_batch(image_batch, label_batch, size):
    import matplotlib.pyplot as plt
    dim = math.ceil(math.sqrt(size))

    plt.figure(figsize=(10, 10))
    for n in range(size):
        ax = plt.subplot(dim, dim, n + 1)
        plt.imshow(image_batch[n])
        plt.title(settings.CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')

    # plt.interactive(False)
    plt.show()
