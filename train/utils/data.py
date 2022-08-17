from train.configs import *

AUTOTUNE = tf.data.AUTOTUNE


def auto_class_weights(dataframe):
    # TODO: Use sklearn class_weight
    __min_count = dataframe.label.value_counts().values.min()
    _class_weights = {S2I_LBL_MAP[_cls]: __min_count / _cnt for _cls, _cnt in dataframe.label.value_counts().items()}
    return _class_weights


# def tf_load_image(img_path, img_shape=(512, 512, 3)):
#     """ Load an image with the correct size and shape """
#     img = tf.image.decode_image(tf.io.read_file(img_path), channels=img_shape[-1])
#     img = tf.reshape(img, img_shape)
#     return img


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SHAPE[0])
    image = tf.cast(image, tf.float32) / 255.
    return image, tf.one_hot(label, N_CLASSES, dtype=tf.uint8)


# def augment(img_batch, label):
#     img_batch = tf.image.random_brightness(img_batch, 0.2)
#     img_batch = tf.image.random_contrast(img_batch, 0.5, 2.0)
#     img_batch = tf.image.random_saturation(img_batch, 0.75, 1.25)
#     img_batch = tf.image.random_hue(img_batch, 0.1)
#     return img_batch, label


def augment(image, label):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    if p_pixel >= .2:
        if p_pixel >= .8:
            image = tf.image.random_saturation(image, lower=.7, upper=1.3)
        elif p_pixel >= .6:
            image = tf.image.random_contrast(image, lower=.8, upper=1.2)
        elif p_pixel >= .4:
            image = tf.image.random_brightness(image, max_delta=.1)
        else:
            image = tf.image.adjust_gamma(image, gamma=.6)

    if p_crop > .7:
        if p_crop > .9:
            image = tf.image.central_crop(image, central_fraction=.6)
        elif p_crop > .8:
            image = tf.image.central_crop(image, central_fraction=.7)
        else:
            image = tf.image.central_crop(image, central_fraction=.8)
    elif p_crop > .4:
        crop_size = tf.random.uniform([], int(IMAGE_SHAPE[0] * .6), IMAGE_SHAPE[0], dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, IMAGE_SHAPE[2]])

    image = tf.image.resize(image, size=IMAGE_SHAPE[:2])
    image = tf.reshape(image, IMAGE_SHAPE)

    return image, label


def create_dataset(df, ordered=False,
                   batch_size=None, drop_remainder=True,
                   cached=True, repeated=True,
                   augmented=True, ):
    dataset = tf.data.Dataset.from_tensor_slices((df.image_path.values, df.label.map(S2I_LBL_MAP).values))
    if not ordered:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    if not ordered:
        dataset = dataset.shuffle(1024)
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if cached:
        dataset = dataset.cache()
    if repeated:
        dataset = dataset.repeat()
    if augmented:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



