try:
    INTERACTIVE
except Exception:
    from generic import *
    from train.setup_configs import *

AUTOTUNE = tf.data.AUTOTUNE


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SHAPE[:2])
    image = tf.cast(image, tf.float32) / 255.
    return image, tf.one_hot(label, N_CLASSES, dtype=tf.uint8)


def simple_augment(img_batch, label):
    img_batch = tf.image.random_brightness(img_batch, 0.2)
    img_batch = tf.image.random_contrast(img_batch, 0.5, 2.0)
    img_batch = tf.image.random_saturation(img_batch, 0.75, 1.25)
    img_batch = tf.image.random_hue(img_batch, 0.1)
    return img_batch, label


def create_dataset(df, ordered=False,
                   batch_size=None, drop_remainder=True,
                   cached=True, repeated=True,
                   augmented=True):
    dataset = tf.data.Dataset.from_tensor_slices((df.image_path.values, df.label.map(S2I_LBL_MAP).values))
    if not ordered:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    if augmented:
        dataset = dataset.map(simple_augment, num_parallel_calls=AUTOTUNE)
    if not ordered:
        dataset = dataset.shuffle(1024)
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if cached:
        dataset = dataset.cache()
    if repeated:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
